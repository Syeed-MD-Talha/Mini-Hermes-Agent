"""
Hermes-style delegate_task tool backed by the delegation manager.

The parent builds a self-contained task packet.  The child agent runs in
isolation with restricted tools and returns a structured result.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .manager import DelegationManager


class SubagentTool:
    """Tool that delegates work to an isolated child agent."""

    def __init__(self, manager: "DelegationManager", default_workspace: str = ".") -> None:
        self.manager = manager
        self.default_workspace = str(Path(default_workspace).resolve())

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": (
                    "Delegate a focused task to an isolated subagent. "
                    "Use for research, coding, review, planning, or skill authoring. "
                    "The subagent runs with its own context and tools and returns a structured result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expertise": {
                            "type": "string",
                            "enum": ["researcher", "coder", "reviewer", "skill-author", "planner", "general"],
                            "description": "Subagent specialization.",
                        },
                        "task": {
                            "type": "string",
                            "description": "Clear, self-contained goal for the subagent.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Background context the subagent needs (files, errors, constraints).",
                        },
                        "relevant_memory": {
                            "type": "string",
                            "description": "Relevant persistent memory to inject into the subagent.",
                        },
                        "relevant_skills": {
                            "type": "string",
                            "description": "Relevant skill instructions to inject into the subagent.",
                        },
                        "constraints": {
                            "type": "string",
                            "description": "Constraints the subagent must respect.",
                        },
                        "workspace": {
                            "type": "string",
                            "description": "Working directory for the subagent. Defaults to the parent's workspace.",
                        },
                    },
                    "required": ["expertise", "task"],
                },
            },
        }

    def run(self, arguments: dict) -> dict:
        expertise = arguments.get("expertise", "general")
        task = arguments.get("task", "")
        context = arguments.get("context", "")
        relevant_memory = arguments.get("relevant_memory", "")
        relevant_skills = arguments.get("relevant_skills", "")
        constraints = arguments.get("constraints", "")
        workspace = arguments.get("workspace") or self.default_workspace

        if not task:
            return {"ok": False, "error": "task is required"}

        # Add expertise-specific guidance to the goal.
        goal = f"[{expertise}] {task}"

        result = self.manager.spawn(
            goal=goal,
            context=context,
            relevant_memory=relevant_memory,
            relevant_skills=relevant_skills,
            constraints=constraints,
            workspace=workspace,
        )

        return {
            "ok": result.status == "completed",
            "task_id": result.task_id,
            "status": result.status,
            "exit_reason": result.exit_reason,
            "summary": result.summary,
            "turns_used": result.turns_used,
            "duration_seconds": result.duration_seconds,
            "tool_trace": result.tool_trace,
            "artifacts": result.artifacts,
        }

