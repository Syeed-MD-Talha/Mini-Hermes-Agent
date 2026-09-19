"""
Hermes-style delegation manager and child agent.

A subagent is a fresh agent instance with an isolated context.  The parent
builds a self-contained task packet (goal + context + selected memory +
selected skills + output contract).  The child runs its own tool loop and
returns a structured result.  Intermediate tool calls do not pollute the
parent conversation.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI


MAX_CHILD_TURNS = 15
CHILD_TIMEOUT_SECONDS = 180
MAX_CHILD_DEPTH = 2
SUMMARY_BUDGET_CHARS = 4_000


@dataclass
class DelegationTask:
    task_id: str
    goal: str
    context: str = ""
    relevant_memory: str = ""
    relevant_skills: str = ""
    constraints: str = ""
    output_schema: Optional[dict] = None
    workspace: str = "."
    parent_task_id: Optional[str] = None
    depth: int = 0


@dataclass
class SubagentResult:
    task_id: str
    status: str  # completed | failed | interrupted | max_iterations
    summary: str
    exit_reason: str
    duration_seconds: float
    turns_used: int
    tool_trace: List[str] = field(default_factory=list)
    raw_output: str = ""
    artifacts: List[str] = field(default_factory=list)


CHILD_SYSTEM_PROMPT = """You are a focused subagent working on a single delegated task.

You do NOT have access to the parent conversation. Use only the context provided below.

WORKSPACE: {workspace}

RELEVANT MEMORY:
{relevant_memory}

RELEVANT SKILLS:
{relevant_skills}

CONSTRAINTS:
{constraints}

OUTPUT CONTRACT:
{output_contract}

RULES:
- Work autonomously. Do not ask the user questions.
- Use the provided tools to complete the task.
- When finished, return a concise final summary of what you did and what the result is.
- If you changed files, list them.
- Do not delegate to other subagents unless explicitly instructed.
"""


class ChildAgent:
    """A fresh agent instance for a single delegated task."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        task: DelegationTask,
        tool_schemas: List[dict],
        tool_executor: Callable[[str, str], dict],
    ) -> None:
        self.client = client
        self.model = model
        self.task = task
        self.tool_schemas = tool_schemas
        self.tool_executor = tool_executor
        self.messages: List[Dict[str, Any]] = []
        self.tool_trace: List[str] = []
        self.artifacts: List[str] = []

    def run(self) -> SubagentResult:
        start = time.time()
        self._build_system_prompt()
        self.messages.append({"role": "user", "content": f"Goal: {self.task.goal}"})

        for turn in range(MAX_CHILD_TURNS):
            elapsed = time.time() - start
            if elapsed > CHILD_TIMEOUT_SECONDS:
                return self._build_result("failed", "timeout", turn, elapsed)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=self.messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    timeout=30,
                )
            except Exception as e:
                return self._build_result("failed", f"api_error: {e}", turn, time.time() - start)

            choice = response.choices[0]
            message = choice.message

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

            if not message.tool_calls:
                return self._build_result(
                    "completed", "completed", turn + 1, time.time() - start
                )

            for tc in message.tool_calls:
                tool_name = tc.function.name
                self.tool_trace.append(tool_name)

                # Block nested delegation at leaf depth.
                if tool_name == "delegate_task" and self.task.depth >= MAX_CHILD_DEPTH:
                    result = {
                        "ok": False,
                        "error": "nested subagent depth limit reached",
                    }
                else:
                    result = self.tool_executor(tool_name, tc.function.arguments)

                # Track created files as artifacts.
                if tool_name in ("write_file", "skill_manage") and result.get("ok"):
                    if "path" in result:
                        self.artifacts.append(result["path"])
                    if "file_path" in result:
                        self.artifacts.append(result["file_path"])

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return self._build_result("failed", "max_iterations", MAX_CHILD_TURNS, time.time() - start)

    def _build_system_prompt(self) -> None:
        output_contract = "Return a concise final summary of what you did and the result."
        if self.task.output_schema:
            output_contract = (
                "Your FINAL response must be ONLY a JSON object validating against this schema:\n"
                f"{json.dumps(self.task.output_schema, indent=2)}"
            )

        system_content = CHILD_SYSTEM_PROMPT.format(
            workspace=self.task.workspace,
            relevant_memory=self.task.relevant_memory or "(none)",
            relevant_skills=self.task.relevant_skills or "(none)",
            constraints=self.task.constraints or "(none)",
            output_contract=output_contract,
        )
        self.messages.append({"role": "system", "content": system_content})

    def _build_result(
        self, status: str, exit_reason: str, turns: int, duration: float
    ) -> SubagentResult:
        raw = self.messages[-1].get("content", "") if self.messages else ""
        summary = raw
        if len(raw) > SUMMARY_BUDGET_CHARS:
            head = raw[: SUMMARY_BUDGET_CHARS // 2]
            tail = raw[-(SUMMARY_BUDGET_CHARS // 2) :]
            summary = f"{head}\n... [truncated] ...\n{tail}"

        return SubagentResult(
            task_id=self.task.task_id,
            status=status,
            summary=summary,
            exit_reason=exit_reason,
            duration_seconds=round(duration, 2),
            turns_used=turns,
            tool_trace=self.tool_trace,
            raw_output=raw,
            artifacts=self.artifacts,
        )


class DelegationManager:
    """Creates and tracks child agents."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        tool_schemas_provider: Callable[[int], List[dict]],
        tool_executor_provider: Callable[[int], Callable[[str, str], dict]],
    ) -> None:
        self.client = client
        self.model = model
        self.tool_schemas_provider = tool_schemas_provider
        self.tool_executor_provider = tool_executor_provider
        self.active: Dict[str, ChildAgent] = {}
        self.results: Dict[str, SubagentResult] = {}

    def spawn(
        self,
        goal: str,
        context: str = "",
        relevant_memory: str = "",
        relevant_skills: str = "",
        constraints: str = "",
        output_schema: Optional[dict] = None,
        workspace: str = ".",
        parent_task_id: Optional[str] = None,
        depth: int = 0,
    ) -> SubagentResult:
        task_id = f"sa-{depth}-{uuid.uuid4().hex[:8]}"
        task = DelegationTask(
            task_id=task_id,
            goal=goal,
            context=context,
            relevant_memory=relevant_memory,
            relevant_skills=relevant_skills,
            constraints=constraints,
            output_schema=output_schema,
            workspace=workspace,
            parent_task_id=parent_task_id,
            depth=depth,
        )

        schemas = self.tool_schemas_provider(depth)
        executor = self.tool_executor_provider(depth)

        child = ChildAgent(self.client, self.model, task, schemas, executor)
        self.active[task_id] = child
        print(f"[delegation] {task_id} started: {goal[:60]}...")

        result = child.run()
        self.active.pop(task_id, None)
        self.results[task_id] = result

        print(
            f"[delegation] {task_id} finished: status={result.status}, "
            f"turns={result.turns_used}, duration={result.duration_seconds}s"
        )
        return result

    def get_result(self, task_id: str) -> Optional[SubagentResult]:
        return self.results.get(task_id)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
