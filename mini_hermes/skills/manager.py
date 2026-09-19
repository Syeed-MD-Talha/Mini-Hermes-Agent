"""
Hermes-style skill_manager tool.

Allows the agent to create, patch, delete, and manage supporting files
for skills.  All writes are security-scanned and use atomic replacement.
"""

from __future__ import annotations

import re
from typing import Optional

from .store import SkillStore


class SkillManagerTool:
    """LLM-facing skill mutation tool."""

    def __init__(self, store: SkillStore) -> None:
        self.store = store

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "skill_manage",
                "description": (
                    "Create or modify reusable procedural knowledge (skills). "
                    "Use when the agent learns a non-trivial workflow that should be reused. "
                    "Do not use for one-off facts (use memory) or temporary task state."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create", "patch", "delete", "write_file", "remove_file"],
                        },
                        "name": {
                            "type": "string",
                            "description": "Skill name (kebab-case).",
                        },
                        "category": {
                            "type": "string",
                            "description": "Category folder (kebab-case). Required for create.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Short routing description (≤60 chars). Required for create.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full SKILL.md content for create or replacement.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact substring to replace for patch.",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement substring for patch.",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Supporting file path for write_file/remove_file.",
                        },
                        "file_content": {
                            "type": "string",
                            "description": "Content for write_file.",
                        },
                    },
                    "required": ["action", "name"],
                },
            },
        }

    def run(self, arguments: dict) -> dict:
        action = arguments.get("action")
        name = arguments.get("name", "")

        if not self.store._is_valid_name(name):
            return {"ok": False, "error": f"invalid skill name: {name}"}

        if action == "create":
            category = arguments.get("category", "general")
            description = arguments.get("description", "")
            content = arguments.get("content", "")
            if not description:
                return {"ok": False, "error": "description required for create"}
            if not content:
                return {"ok": False, "error": "content required for create"}
            if self._is_suspicious(content):
                return {"ok": False, "error": "skill content failed security scan"}
            return self.store.create_skill(name, category, description, content)

        if action == "patch":
            old_string = arguments.get("old_string", "")
            new_string = arguments.get("new_string", "")
            if not old_string:
                return {"ok": False, "error": "old_string required for patch"}
            if self._is_suspicious(new_string):
                return {"ok": False, "error": "patch content failed security scan"}
            return self.store.patch_skill(name, old_string, new_string)

        if action == "delete":
            return self.store.delete_skill(name, hard=False)

        if action == "write_file":
            file_path = arguments.get("file_path", "")
            file_content = arguments.get("file_content", "")
            if not file_path:
                return {"ok": False, "error": "file_path required"}
            if self._is_suspicious(file_content):
                return {"ok": False, "error": "file content failed security scan"}
            return self.store.write_supporting_file(name, file_path, file_content)

        if action == "remove_file":
            file_path = arguments.get("file_path", "")
            if not file_path:
                return {"ok": False, "error": "file_path required"}
            return self.store.remove_supporting_file(name, file_path)

        return {"ok": False, "error": f"unknown action: {action}"}

    @staticmethod
    def _is_suspicious(text: str) -> bool:
        lowered = text.lower()
        patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"you are now",
            r"disregard.*above",
            r"rm -rf /",
            r"format.*disk",
            r"<!--",
            r"<script",
        ]
        return any(re.search(p, lowered) for p in patterns)
