"""
Mini-Hermes agent with layered memory.

Architecture:
  - Curated persistent memory: ~/.mini-hermes/memories/{MEMORY,USER}.md
  - Episodic session archive:  ~/.mini-hermes/state.db (FTS5)
  - Context compression:       head + structured summary + tail
  - Tools: memory, session_search
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from context_compressor import ContextCompressor, estimate_tokens
from memory_store import MemoryStore
from memory_tool import MemoryTool
from session_search import SessionSearchTool
from session_store import SessionStore


SYSTEM_PROMPT_TEMPLATE = """You are a helpful coding assistant with persistent memory.

{memory_section}

{user_section}

You have access to tools. Use them when needed. When you learn durable facts about the user or environment, call the `memory` tool. When the user refers to past conversations, call `session_search`.

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
        self.memory_tool = MemoryTool(self.memory_store)
        self.session_search_tool = SessionSearchTool(self.session_store)
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

        system_content = SYSTEM_PROMPT_TEMPLATE.format(
            memory_section=memory_section,
            user_section=user_section,
            current_date=date.today().isoformat(),
        )
        self.messages.append({"role": "system", "content": system_content})

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run_interactive(self) -> None:
        self.start_session()
        print(f"Using Fireworks model: {self.model}")
        print("Type your question (or 'exit' to quit).\n")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            self._add_message("user", user_input)
            self._maybe_compress()
            self._respond()

    # ------------------------------------------------------------------
    # Response handling with tool support
    # ------------------------------------------------------------------
    def _respond(self) -> None:
        tools = [self.memory_tool.schema(), self.session_search_tool.schema()]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2048,
                messages=self.messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            print(f"\nError: {e}\n")
            return

        choice = response.choices[0]
        message = choice.message

        # Record assistant message (with tool calls if any).
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if message.content:
            assistant_msg["content"] = message.content
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        self.messages.append(assistant_msg)
        self._persist_message("assistant", message.content, tool_calls=assistant_msg.get("tool_calls"))

        if not message.tool_calls:
            print(f"\nAssistant: {message.content or '(no content)'}\n")
            return

        # Execute tools and continue the loop.
        tool_results: List[Dict[str, Any]] = []
        for tc in message.tool_calls:
            result = self._execute_tool(tc.function.name, tc.function.arguments)
            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
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

    def _execute_tool(self, name: str, arguments_json: str) -> dict:
        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError:
            return {"ok": False, "error": "invalid JSON arguments"}

        if name == "memory":
            return self.memory_tool.run(arguments)
        if name == "session_search":
            return self.session_search_tool.run(arguments)
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
