"""
Hermes-style curated memory store.

Stores two small plain-text files:
  ~/.mini-hermes/memories/MEMORY.md  -> agent/environment knowledge
  ~/.mini-hermes/memories/USER.md    -> user profile/preferences

Entries are separated by '\n§\n'.  Writes are atomic, budget-enforced,
and use file locking.  A frozen snapshot is captured at session start
and is NOT rebuilt mid-session (preserves prompt-cache stability).
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]


DEFAULT_MEMORY_BUDGET = 2200
DEFAULT_USER_BUDGET = 1375
ENTRY_SEPARATOR = "\n§\n"


@dataclass
class MemoryState:
    memory_entries: List[str] = field(default_factory=list)
    user_entries: List[str] = field(default_factory=list)


class MemoryStore:
    """Curated persistent memory with atomic writes and budget enforcement."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        memory_budget: int = DEFAULT_MEMORY_BUDGET,
        user_budget: int = DEFAULT_USER_BUDGET,
    ) -> None:
        self.base_dir = base_dir or Path.home() / ".mini-hermes"
        self.memories_dir = self.base_dir / "memories"
        self.memories_dir.mkdir(parents=True, exist_ok=True)

        self.memory_path = self.memories_dir / "MEMORY.md"
        self.user_path = self.memories_dir / "USER.md"
        self.memory_lock_path = self.memories_dir / "MEMORY.md.lock"
        self.user_lock_path = self.memories_dir / "USER.md.lock"

        self.memory_budget = memory_budget
        self.user_budget = user_budget

        # Live mutable state.
        self._state = MemoryState()
        # Frozen snapshot captured at session start.
        self._snapshot = MemoryState()

        self._load()
        self._snapshot = MemoryState(
            memory_entries=list(self._state.memory_entries),
            user_entries=list(self._state.user_entries),
        )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------
    def get_snapshot_text(self) -> tuple[str, str]:
        """Return the frozen system-prompt snapshot."""
        return self._render(self._snapshot.memory_entries), self._render(
            self._snapshot.user_entries
        )

    def get_live_text(self) -> tuple[str, str]:
        """Return the live on-disk state (for tool results / debugging)."""
        return self._render(self._state.memory_entries), self._render(
            self._state.user_entries
        )

    def get_entries(self, target: str) -> List[str]:
        if target == "memory":
            return list(self._state.memory_entries)
        if target == "user":
            return list(self._state.user_entries)
        raise ValueError(f"target must be 'memory' or 'user', got {target}")

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------
    def add(self, target: str, content: str) -> dict:
        """Add a single entry. Returns status dict for the LLM."""
        return self.apply_batch(target, [{"action": "add", "content": content}])

    def replace(
        self, target: str, old_text: str, content: str, *, strict: bool = True
    ) -> dict:
        """Replace an entry containing old_text with new content."""
        return self.apply_batch(
            target, [{"action": "replace", "old_text": old_text, "content": content}],
            strict=strict,
        )

    def remove(self, target: str, old_text: str, *, strict: bool = True) -> dict:
        """Remove an entry containing old_text."""
        return self.apply_batch(
            target, [{"action": "remove", "old_text": old_text}], strict=strict
        )

    def apply_batch(
        self, target: str, operations: Iterable[dict], *, strict: bool = True
    ) -> dict:
        """
        Apply a batch of add/replace/remove operations atomically.
        The final state must fit the budget or the whole batch is rejected.
        """
        if target not in ("memory", "user"):
            return {"ok": False, "error": f"invalid target: {target}"}

        budget = self.memory_budget if target == "memory" else self.user_budget
        entries = list(self._state.memory_entries if target == "memory" else self._state.user_entries)

        # Build a working copy.
        working = list(entries)
        for op in operations:
            action = op.get("action")
            if action == "add":
                working.append(op["content"].strip())
            elif action == "replace":
                idx = self._find_unique_entry(working, op["old_text"], strict=strict)
                if idx is None:
                    return {
                        "ok": False,
                        "error": f"replace target not found or ambiguous: {op.get('old_text')!r}",
                    }
                working[idx] = op["content"].strip()
            elif action == "remove":
                idx = self._find_unique_entry(working, op["old_text"], strict=strict)
                if idx is None:
                    return {
                        "ok": False,
                        "error": f"remove target not found or ambiguous: {op.get('old_text')!r}",
                    }
                del working[idx]
            else:
                return {"ok": False, "error": f"unknown action: {action}"}

        final_text = self._render(working)
        if len(final_text) > budget:
            return {
                "ok": False,
                "error": (
                    f"batch would exceed {target} budget ({len(final_text)}/{budget} chars). "
                    "Consolidate or remove existing entries first."
                ),
                "current_entries": entries,
                "current_length": len(self._render(entries)),
                "budget": budget,
            }

        # Threat-scan before persisting.
        for entry in working:
            if self._is_suspicious(entry):
                return {
                    "ok": False,
                    "error": "suspicious memory entry blocked (possible prompt injection)",
                    "entry": entry,
                }

        # Commit to disk atomically.
        path = self.memory_path if target == "memory" else self.user_path
        lock_path = self.memory_lock_path if target == "memory" else self.user_lock_path
        self._atomic_write(path, final_text, lock_path)

        # Update live state (snapshot stays frozen).
        if target == "memory":
            self._state.memory_entries = working
        else:
            self._state.user_entries = working

        return {
            "ok": True,
            "target": target,
            "length": len(final_text),
            "budget": budget,
            "entries": working,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        self._state.memory_entries = self._parse_file(self.memory_path)
        self._state.user_entries = self._parse_file(self.user_path)

    @staticmethod
    def _parse_file(path: Path) -> List[str]:
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        # Normalize legacy separators.
        text = text.replace("\r\n", "\n")
        parts = [p.strip() for p in text.split("§")]
        return [p for p in parts if p]

    @staticmethod
    def _render(entries: List[str]) -> str:
        if not entries:
            return ""
        return ENTRY_SEPARATOR.join(entries)

    @staticmethod
    def _find_unique_entry(entries: List[str], substring: str, *, strict: bool = True) -> Optional[int]:
        substring = substring.strip()
        matches = [i for i, e in enumerate(entries) if substring in e]
        if len(matches) == 1:
            return matches[0]
        if not strict and matches:
            return matches[0]
        return None

    @staticmethod
    def _is_suspicious(text: str) -> bool:
        """Basic prompt-injection / exfiltration heuristics."""
        lowered = text.lower()
        patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"you are now",
            r"disregard.*above",
            r"<!--",
            r"\{\{.*\}\}",
            r"<script",
        ]
        return any(re.search(p, lowered) for p in patterns)

    @staticmethod
    def _atomic_write(path: Path, text: str, lock_path: Path) -> None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                except (OSError, AttributeError):
                    pass
            fd, tmp = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                shutil.move(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except FileNotFoundError:
                    pass
                raise
