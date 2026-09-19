"""
Hermes-style memory tool exposed to the LLM.

Supports add / replace / remove and batch operations against the
curated persistent memory store (MEMORY.md / USER.md).
"""

from __future__ import annotations

from typing import Any, Dict, List

from memory_store import MemoryStore


MEMORY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": (
            "Save durable facts that survive across sessions. "
            "Use for user identity, stable preferences, standing conventions, "
            "and stable environment facts. "
            "Do NOT use for task progress, temporary TODOs, completed work logs, "
            "raw data, easily rediscovered facts, or reusable procedures that belong in skills."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["user", "memory"],
                    "description": "'user' for user profile, 'memory' for agent/environment knowledge.",
                },
                "action": {
                    "type": "string",
                    "enum": ["add", "replace", "remove"],
                    "description": "Single action. Use 'operations' for batches.",
                },
                "content": {
                    "type": "string",
                    "description": "Content for add/replace.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Unique substring of the entry to replace or remove.",
                },
                "operations": {
                    "type": "array",
                    "description": "Batch of add/replace/remove operations applied atomically.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                            "content": {"type": "string"},
                            "old_text": {"type": "string"},
                        },
                        "required": ["action"],
                    },
                },
            },
            "required": ["target"],
        },
    },
}


class MemoryTool:
    """LLM-facing memory tool."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def schema(self) -> dict:
        return MEMORY_TOOL_SCHEMA

    def run(self, arguments: dict) -> dict:
        target = arguments.get("target")
        if target not in ("user", "memory"):
            return {"ok": False, "error": "target must be 'user' or 'memory'"}

        operations = arguments.get("operations")
        if operations:
            return self.store.apply_batch(target, operations)

        action = arguments.get("action")
        content = arguments.get("content", "")
        old_text = arguments.get("old_text", "")

        if action == "add":
            return self.store.add(target, content)
        if action == "replace":
            return self.store.replace(target, old_text, content)
        if action == "remove":
            return self.store.remove(target, old_text)

        return {"ok": False, "error": f"unknown action: {action}"}
