"""
Mini-Hermes agent with layered memory and skills.

Architecture:
  - Curated persistent memory: ~/.mini-hermes/memories/{MEMORY,USER}.md
  - Episodic session archive:  ~/.mini-hermes/state.db (FTS5)
  - Procedural memory (skills): ~/.mini-hermes/skills/
  - Context compression:       head + structured summary + tail
  - Tools: memory, session_search, skills_list, skill_view, skill_manage, delegate_task, read_file, write_file, search_files, patch, execute_code
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from mini_hermes.context import ContextCompressor, estimate_tokens
from mini_hermes.delegation import DelegationManager, SubagentTool
from mini_hermes.learn import build_learn_prompt, is_learn_command
from mini_hermes.memory import MemoryStore, MemoryTool
from mini_hermes.sessions import SessionSearchTool, SessionStore
from mini_hermes.skills import SkillManagerTool, SkillStore, SkillsTool
from mini_hermes.tools import WebTools, WorkspaceTools


SYSTEM_PROMPT_TEMPLATE = """You are a helpful coding assistant with persistent memory, reusable skills, and subagents.

{memory_section}

{user_section}

## AVAILABLE SKILLS (compact index)
Load a skill with skill_view(name) when relevant. Supporting files can be loaded with skill_view(name, file_path).

{skills_index}

## WORKFLOW RULES — FOLLOW STRICTLY
1. You DO have file-write tools. NEVER claim otherwise. Use write_file to create files and patch to edit them.
2. When the user asks for code, functions, scripts, tests, or project changes, you MUST create or modify files in the current workspace. NEVER return code only in chat. NEVER ask the user to "drop it into a folder" — you are the one who writes files.
3. For coding tasks, first delegate to the `coder` subagent via delegate_task(expertise="coder", task=...). Give it the full task and context. The coder subagent will write files and run tests.
4. After the coder subagent returns, verify the files exist by reading them, then summarize what was created for the user.
5. For research before coding, use the `researcher` subagent via delegate_task(expertise="researcher", task=...).
6. For code review, use the `reviewer` subagent via delegate_task(expertise="reviewer", task=...).
7. For planning complex multi-step work, use the `planner` subagent via delegate_task(expertise="planner", task=...).
8. For turning a successful workflow into a reusable skill, use the `skill-author` subagent via delegate_task(expertise="skill-author", task=...).
9. When you learn durable facts about the user or environment, call the `memory` tool.
10. When the user refers to past conversations, call `session_search`.
11. When a reusable procedure applies, call `skills_list` / `skill_view` / `skill_manage`.
12. **AUTO-SKILL CREATION**: If no existing skill matches the user's task, or if the task would benefit from a reusable procedure, first use `skill_manage(action="create", ...)` to author a skill, then execute it. Treat skill creation as a normal tool step, not a separate mode.
13. For creating a new skill, delegate to the `skill-author` subagent via `delegate_task(expertise="skill-author", task=...)`. The skill-author will write a complete SKILL.md with examples and constraints.
14. **WEB SEARCH**: For current events, weather, news, sports scores, stock prices, or any time-sensitive facts, you MUST use `web_search`. Do not rely on training data. After searching, use `web_fetch` on the most promising result to verify details before answering.
15. **TIME QUESTIONS**: For "what time is it" or any timezone/time question, use `get_time(timezone=...)`. Do NOT use web search for time. Common timezones: UTC, Asia/Dhaka, America/New_York, Europe/London.
16. **HONESTY ABOUT DATA FRESHNESS**: Always report when the data was published or observed. If you cannot find current data, say so instead of guessing. Never invent exact numbers, times, or conditions.

## FEW-SHOT EXAMPLES

### Example 1: User asks for code
User: "Write a function to validate email addresses."
Correct response: call delegate_task(expertise="coder", task="Write a Python function to validate email addresses using regex. Save it as email_utils.py in the current workspace and create pytest tests in test_email_utils.py. Run the tests and report results.")
Incorrect response: returning code in chat.

### Example 2: User asks for a script
User: "Create a script that downloads a file from a URL."
Correct response: call delegate_task(expertise="coder", task="Create a Python script download_file.py that downloads a file from a given URL. Include argparse, error handling, and a small test.")
Incorrect response: showing the script in chat.

### Example 3: User asks for tests
User: "Add tests for the email validator."
Correct response: call delegate_task(expertise="coder", task="Add pytest tests for email_utils.py in test_email_utils.py. Cover valid and invalid cases. Run pytest and report results.")
Incorrect response: listing test cases in chat.

### Example 4: User asks for a project change
User: "Refactor main.py to use a class."
Correct response: call delegate_task(expertise="coder", task="Refactor main.py to use a class-based structure. Preserve existing behavior. Run the script to verify it still works.")
Incorrect response: describing the refactor in chat.

### Example 5: User asks for something with no matching skill
User: "Create a PowerPoint presentation about Rust String vs &str."
Correct response:
1. call skills_list to check for a pptx skill.
2. If none exists, call delegate_task(expertise="skill-author", task="Create a skill named 'create-pptx' in category 'productivity' that teaches how to build PowerPoint decks with python-pptx. Include a complete runnable example and constraints.")
3. Then call delegate_task(expertise="coder", task="Use the create-pptx skill to create a PowerPoint about Rust String vs &str. Save it as Rust_String_vs_str.pptx in the workspace.")
Incorrect response: returning a truncated script in chat or claiming you cannot create files.

Current date: {current_date}
"""

MEMORY_SECTION_TEMPLATE = """## AGENT MEMORY (durable environment knowledge)
{memory_text}
"""

USER_SECTION_TEMPLATE = """## USER PROFILE (durable user knowledge)
{user_text}
"""

MODEL_MAX_TOKENS = 128000  # fallback context window estimate
COMPRESSION_THRESHOLD = 0.45


class Agent:
    """Main agent loop with memory and context compression."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("FIREWORKS_API_KEY")
        model = os.getenv("FIREWORKS_MODEL")
        if not api_key or not model:
            print("Error: FIREWORKS_API_KEY and FIREWORKS_MODEL must be set in .env")
            sys.exit(1)

        self.client = OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
        self.model = model

        self.memory_store = MemoryStore()
        self.session_store = SessionStore()
        self.skill_store = SkillStore(project_dir=Path.cwd())
        # Use a dedicated workspace subfolder so generated files don't pollute the agent source.
        self.workspace_dir = Path.cwd() / "workspace"
        self.workspace_dir.mkdir(exist_ok=True)
        self.workspace_tools = WorkspaceTools(str(self.workspace_dir))
        self.web_tools = WebTools()
        self.memory_tool = MemoryTool(self.memory_store)
        self.session_search_tool = SessionSearchTool(self.session_store)
        self.skills_tool = SkillsTool(self.skill_store)
        self.skill_manager_tool = SkillManagerTool(self.skill_store)
        self.delegation_manager = DelegationManager(
            self.client,
            self.model,
            tool_schemas_provider=self._get_child_tool_schemas,
            tool_executor_provider=self._get_child_tool_executor,
        )
        self.subagent_tool = SubagentTool(self.delegation_manager, default_workspace=str(self.workspace_dir))
        self.compressor = ContextCompressor(
            self.client,
            self.model,
            threshold=COMPRESSION_THRESHOLD,
        )

        self.session_id: Optional[str] = None
        self.messages: List[Dict[str, Any]] = []
        self.pending_tool_calls: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def start_session(self) -> None:
        self.session_id = self.session_store.create_session()
        memory_text, user_text = self.memory_store.get_snapshot_text()

        memory_section = (
            MEMORY_SECTION_TEMPLATE.format(memory_text=memory_text)
            if memory_text.strip()
            else "## AGENT MEMORY\n(none yet)"
        )
        user_section = (
            USER_SECTION_TEMPLATE.format(user_text=user_text)
            if user_text.strip()
            else "## USER PROFILE\n(none yet)"
        )
        skills_index = self.skill_store.get_index_text()

        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            memory_section=memory_section,
            user_section=user_section,
            skills_index=skills_index,
            current_date=date.today().isoformat(),
        )
        self.messages.append({"role": "system", "content": system_content})

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_interactive(self) -> None:
        self.start_session()
        print(f"Using Fireworks model: {self.model}")
        print("Type your question, /learn <source>, or 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            is_learn, learn_source = is_learn_command(user_input)
            if is_learn:
                self._enter_learn_mode(learn_source)
                continue

            self._add_message("user", user_input)
            self._maybe_compress()
            self._respond()

    def _enter_learn_mode(self, source: Optional[str]) -> None:
        if not source:
            print("Usage: /learn <source or experience description>\n")
            return
        prompt = build_learn_prompt(source)
        self._add_message("user", prompt)
        print("[entering /learn mode]\n")
        self._respond()

    # ------------------------------------------------------------------
    # Response handling with tool support
    # ------------------------------------------------------------------
    def _respond(self) -> None:
        tools = self._get_tool_schemas()

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=self.messages,
                tools=tools,
                tool_choice="auto",
                stream=True,
            )
        except Exception as e:
            print(f"\nError: {e}\n")
            return

        # Stream the response, collecting content and tool calls.
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        content_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        printed_any = False

        print("\nAssistant: ", end="", flush=True)
        for chunk in stream:
            if not chunk.choices:
                # Some providers emit empty heartbeat/finish chunks.
                continue
            delta = chunk.choices[0].delta

            # Stream text content, suppressing leading blank lines.
            if delta.content:
                text = delta.content
                if not printed_any:
                    text = text.lstrip("\n")
                    if text:
                        printed_any = True
                else:
                    printed_any = True
                if printed_any:
                    print(text, end="", flush=True)
                    content_parts.append(delta.content)

            # Accumulate tool calls (not streamed character-by-character).
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc.id or "",
                            "type": tc.type or "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.type:
                        tool_calls[idx]["type"] = tc.type
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["function"]["arguments"] += tc.function.arguments

        print("\n")

        content = "".join(content_parts)
        if content:
            assistant_msg["content"] = content
        if tool_calls:
            assistant_msg["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

        self.messages.append(assistant_msg)
        self._persist_message("assistant", content, tool_calls=assistant_msg.get("tool_calls"))

        if not tool_calls:
            return

        # Execute tools and continue the loop.
        tool_results: List[Dict[str, Any]] = []
        for tc in assistant_msg["tool_calls"]:
            result = self._execute_tool(tc["function"]["name"], tc["function"]["arguments"])
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

        for tr in tool_results:
            self.messages.append(tr)
            self._persist_message(
                "tool",
                tr["content"],
                tool_call_id=tr["tool_call_id"],
                name=tr["name"],
            )

        # Recurse to get final answer after tool results.
        self._respond()

    def _get_tool_schemas(self) -> List[dict]:
        return [
            self.memory_tool.schema(),
            self.session_search_tool.schema(),
            self.skills_tool.list_schema(),
            self.skills_tool.view_schema(),
            self.skill_manager_tool.schema(),
            self.subagent_tool.schema(),
            *self.web_tools.schemas(),
            *self.workspace_tools.schemas(),
        ]

    def _get_child_tool_schemas(self, depth: int) -> List[dict]:
        """Tools available to a child subagent. Leaf children cannot delegate further."""
        schemas = [
            self.memory_tool.schema(),
            self.session_search_tool.schema(),
            self.skills_tool.list_schema(),
            self.skills_tool.view_schema(),
            self.skill_manager_tool.schema(),
            *self.web_tools.schemas(),
            *self.workspace_tools.schemas(),
        ]
        # Only non-leaf subagents can delegate.
        if depth < 1:
            schemas.append(self.subagent_tool.schema())
        return schemas

    def _get_child_tool_executor(self, depth: int):
        """Return a tool executor bound to the given subagent depth."""
        def executor(name: str, arguments_json: str) -> dict:
            if name == "delegate_task" and depth >= 1:
                return {"ok": False, "error": "leaf subagent cannot delegate further"}
            return self._execute_tool(name, arguments_json)
        return executor

    def _execute_tool(self, name: str, arguments_json: str) -> dict:
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            return {"ok": False, "error": "invalid JSON arguments"}

        if name == "memory":
            return self.memory_tool.run(arguments)
        if name == "session_search":
            return self.session_search_tool.run(arguments)
        if name == "skills_list":
            return self.skills_tool.list()
        if name == "skill_view":
            return self.skills_tool.view(arguments.get("name"), arguments.get("file_path"))
        if name == "skill_manage":
            return self.skill_manager_tool.run(arguments)
        if name == "delegate_task":
            return self.subagent_tool.run(arguments)
        if name in ("web_search", "web_fetch"):
            return self.web_tools.execute(name, arguments)
        if name in ("read_file", "write_file", "search_files", "patch", "execute_code", "get_time"):
            return self.workspace_tools.execute(name, arguments)
        return {"ok": False, "error": f"unknown tool: {name}"}

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------
    def _maybe_compress(self) -> None:
        if not self.compressor.should_compress(self.messages, MODEL_MAX_TOKENS):
            return

        memory_context = "\n".join(self.memory_store.get_snapshot_text())
        # Try to preserve the latest user request by focusing on it.
        focus = ""
        for msg in reversed(self.messages):
            if msg.get("role") == "user" and msg.get("content"):
                focus = str(msg["content"])[:200]
                break

        result = self.compressor.compress(
            self.messages,
            memory_context=memory_context,
            focus=focus,
        )
        if result.messages is not self.messages:
            print(
                f"[context compressed: {len(self.messages)} -> {len(result.messages)} messages, "
                f"summary {len(result.summary)} chars]"
            )
            self.messages = result.messages

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._persist_message(role, content)

    def _persist_message(
        self,
        role: str,
        content: Optional[str],
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        if self.session_id is None:
            return
        self.session_store.append_message(
            self.session_id,
            role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
        )


def main() -> None:
    agent = Agent()
    agent.run_interactive()


if __name__ == "__main__":
    main()
