"""
Hermes-style /learn prompt builder.

/learn is not a separate ingestion pipeline. It builds a standards-guided
prompt and hands it to the normal agent turn. The agent then uses its normal
tools (read, search, web) to gather source material and writes the resulting
skill via skill_manage.
"""

from __future__ import annotations

from typing import Optional


LEARN_PROMPT_TEMPLATE = """You are in /learn mode. The user wants you to turn the following source or experience into a reusable skill (procedural memory).

Source/experience:
{source}

Instructions:
1. First call skills_list to check whether an existing skill already covers this topic. If one exists, prefer patching it rather than creating a duplicate.
2. Gather the source material using your normal tools if needed (read files, search, web_extract, etc.).
3. Distill reusable procedural knowledge, not a narrative of what happened. Remove incident numbers, dates, and one-off details. Preserve the generalized rule.
4. Write the skill using skill_manage(action="create" or "patch"). Use kebab-case names and categories.
5. Keep SKILL.md focused and operational (~100 lines simple, ~200 lines complex). Move large reference material into references/ files.
6. Structure SKILL.md with:
   - YAML frontmatter: name, description, version
   - # Title
   - ## Overview
   - ## When to Use
   - ## Prerequisites
   - ## Quick Reference
   - ## Procedure
   - ## Pitfalls
   - ## Verification
7. Frame actions in terms of your available tools (terminal, read_file, write_file, search_files, patch, etc.) rather than generic human instructions.
8. Do not invent flags, endpoints, or APIs. Use exact commands and config keys from the source.
9. If the source is large (book, docs), create a knowledge-base skill: a short index/mental-model SKILL.md plus references/<chapter>.md files.
10. After writing, call skill_view to verify the skill is readable.

Skill description must be ≤60 characters and immediately communicate when to load this skill.
"""


def build_learn_prompt(source: str) -> str:
    return LEARN_PROMPT_TEMPLATE.format(source=source)


def is_learn_command(text: str) -> tuple[bool, Optional[str]]:
    """Detect /learn command and return its argument."""
    stripped = text.strip()
    if stripped.lower().startswith("/learn"):
        remainder = stripped[6:].strip()
        return True, remainder or None
    return False, None
