"""
Hermes-style skills discovery and viewing tools.

skills_list  -> compact index for routing
skill_view   -> progressive disclosure (SKILL.md or supporting file)
"""

from __future__ import annotations

from typing import Optional

from .store import SkillStore


class SkillsTool:
    """LLM-facing skill discovery and viewing."""

    def __init__(self, store: SkillStore) -> None:
        self.store = store

    def list_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "skills_list",
                "description": (
                    "List available reusable skills. Returns only name and short description. "
                    "Use this to decide which skill to load with skill_view."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    def view_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "skill_view",
                "description": (
                    "Load a skill's full instructions or a supporting file. "
                    "First call without file_path to get SKILL.md and linked files. "
                    "Then call with file_path to load references/templates/scripts/assets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Skill name.",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional supporting file path relative to the skill directory.",
                        },
                    },
                    "required": ["name"],
                },
            },
        }

    def list(self) -> dict:
        return {
            "ok": True,
            "skills": [
                {
                    "name": e.name,
                    "category": e.category,
                    "description": e.description,
                }
                for e in self.store.list_skills()
            ],
        }

    def view(self, name: str, file_path: Optional[str] = None) -> dict:
        result = self.store.view_skill(name, file_path)
        if result.get("ok"):
            self.store.record_use(name)
        return result
