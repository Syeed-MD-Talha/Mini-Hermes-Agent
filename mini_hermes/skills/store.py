"""
Hermes-style skill storage with progressive disclosure.

A skill is a package:

  ~/.mini-hermes/skills/<category>/<skill-name>/
      SKILL.md
      references/
      templates/
      scripts/
      assets/

Only a compact index (name + description) is injected into the system prompt.
The full skill is loaded on demand via skill_view, and supporting files are
loaded only when explicitly requested.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SKILL_DIR_NAME = "skills"
USAGE_FILE = ".usage.json"
ARCHIVE_DIR = ".archive"
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets", "examples"}
MAX_SKILL_MD_CHARS = 100_000
MAX_SUPPORTING_FILE_BYTES = 1 * 1024 * 1024


@dataclass
class SkillEntry:
    name: str
    category: str
    description: str
    path: Path
    metadata: dict = field(default_factory=dict)


class SkillStore:
    """Discovers, reads, and manages skills on disk.

    Supports layered skill resolution:
      1. Project-local: ./.mini-hermes/skills/  (highest precedence)
      2. Global:        ~/.mini-hermes/skills/  (fallback)

    New skills are created in the project-local layer by default.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        project_dir: Optional[Path] = None,
    ) -> None:
        self.global_dir = (base_dir or Path.home() / ".mini-hermes") / SKILL_DIR_NAME
        self.project_dir = (project_dir or Path.cwd() / ".mini-hermes") / SKILL_DIR_NAME

        # Ensure directories exist.
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.archive_dir = self.project_dir / ARCHIVE_DIR
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.usage_path = self.project_dir / USAGE_FILE

        self._index: Dict[str, SkillEntry] = {}
        self._usage: Dict[str, dict] = {}
        self._load_usage()
        self._build_index()

    # ------------------------------------------------------------------
    # Discovery / index
    # ------------------------------------------------------------------
    def list_skills(self, include_archive: bool = False) -> List[SkillEntry]:
        entries = list(self._index.values())
        if not include_archive:
            entries = [e for e in entries if self._usage.get(e.name, {}).get("state") != "archived"]
        return sorted(entries, key=lambda e: (e.category, e.name))

    def get_skill(self, name: str) -> Optional[SkillEntry]:
        return self._index.get(name)

    def get_index_text(self, max_description_len: int = 60) -> str:
        """Compact index for system prompt."""
        lines = []
        for entry in self.list_skills():
            desc = entry.description
            if len(desc) > max_description_len:
                desc = desc[: max_description_len - 3] + "..."
            lines.append(f"- {entry.name}: {desc}")
        if not lines:
            return "(no skills installed)"
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Reading (progressive disclosure)
    # ------------------------------------------------------------------
    def view_skill(self, name: str, file_path: Optional[str] = None) -> dict:
        entry = self._index.get(name)
        if entry is None:
            return {"ok": False, "error": f"skill not found: {name}"}

        if file_path:
            return self._view_supporting_file(entry, file_path)

        skill_md = entry.path / "SKILL.md"
        if not skill_md.exists():
            return {"ok": False, "error": f"SKILL.md missing for {name}"}

        content = skill_md.read_text(encoding="utf-8", errors="replace")
        linked_files = self._list_supporting_files(entry)
        self._record_view(name)
        return {
            "ok": True,
            "name": name,
            "category": entry.category,
            "description": entry.description,
            "metadata": entry.metadata,
            "content": content,
            "linked_files": linked_files,
        }

    def _view_supporting_file(self, entry: SkillEntry, file_path: str) -> dict:
        # Security: prevent directory traversal.
        requested = (entry.path / file_path).resolve()
        if not str(requested).startswith(str(entry.path.resolve())):
            return {"ok": False, "error": "invalid file path"}
        if not requested.exists() or not requested.is_file():
            return {"ok": False, "error": f"file not found: {file_path}"}
        if requested.stat().st_size > MAX_SUPPORTING_FILE_BYTES:
            return {"ok": False, "error": f"file too large: {file_path}"}
        content = requested.read_text(encoding="utf-8", errors="replace")
        return {
            "ok": True,
            "name": entry.name,
            "file_path": file_path,
            "content": content,
        }

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def create_skill(
        self,
        name: str,
        category: str,
        description: str,
        content: str,
        created_by: str = "agent",
        layer: Optional[str] = None,
    ) -> dict:
        if not self._is_valid_name(name):
            return {"ok": False, "error": f"invalid skill name: {name}"}
        if not self._is_valid_name(category):
            return {"ok": False, "error": f"invalid category: {category}"}
        if len(content) > MAX_SKILL_MD_CHARS:
            return {"ok": False, "error": "SKILL.md exceeds 100,000 character limit"}

        # Default to project-local layer for new skills.
        target_layer = self.project_dir if layer != "global" else self.global_dir
        skill_path = target_layer / category / name

        # Prevent shadowing an existing skill in the other layer silently.
        other_layer = self.global_dir if target_layer == self.project_dir else self.project_dir
        if skill_path.exists() or (other_layer / category / name).exists():
            return {"ok": False, "error": f"skill already exists: {name}"}

        skill_path.mkdir(parents=True, exist_ok=True)
        (skill_path / "SKILL.md").write_text(content, encoding="utf-8")

        self._record_creation(name, category, created_by=created_by)
        self._build_index()
        return {"ok": True, "name": name, "path": str(skill_path), "layer": "project" if target_layer == self.project_dir else "global"}

    def patch_skill(self, name: str, old_string: str, new_string: str) -> dict:
        entry = self._index.get(name)
        if entry is None:
            return {"ok": False, "error": f"skill not found: {name}"}

        skill_md = entry.path / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")
        if old_string not in content:
            return {"ok": False, "error": "old_string not found in SKILL.md"}
        if content.count(old_string) > 1:
            return {"ok": False, "error": "old_string is ambiguous (multiple matches)"}

        new_content = content.replace(old_string, new_string, 1)
        if len(new_content) > MAX_SKILL_MD_CHARS:
            return {"ok": False, "error": "patch would exceed 100,000 character limit"}

        self._atomic_write(skill_md, new_content)
        self._record_patch(name)
        self._build_index()
        return {"ok": True, "name": name}

    def delete_skill(self, name: str, hard: bool = False) -> dict:
        entry = self._index.get(name)
        if entry is None:
            return {"ok": False, "error": f"skill not found: {name}"}

        if hard:
            shutil.rmtree(entry.path)
            self._usage.pop(name, None)
            self._save_usage()
        else:
            archive_target = self.archive_dir / entry.category / name
            archive_target.parent.mkdir(parents=True, exist_ok=True)
            if archive_target.exists():
                shutil.rmtree(archive_target)
            shutil.move(str(entry.path), str(archive_target))
            self._set_state(name, "archived")

        self._build_index()
        return {"ok": True, "name": name, "hard": hard}

    def write_supporting_file(
        self, name: str, file_path: str, file_content: str
    ) -> dict:
        entry = self._index.get(name)
        if entry is None:
            return {"ok": False, "error": f"skill not found: {name}"}

        # Only allowed subdirectories.
        parts = Path(file_path).parts
        if not parts or parts[0] not in ALLOWED_SUBDIRS:
            return {
                "ok": False,
                "error": f"file_path must be under one of {ALLOWED_SUBDIRS}",
            }

        target = entry.path / file_path
        resolved = target.resolve()
        if not str(resolved).startswith(str(entry.path.resolve())):
            return {"ok": False, "error": "invalid file path"}

        if len(file_content.encode("utf-8")) > MAX_SUPPORTING_FILE_BYTES:
            return {"ok": False, "error": "supporting file exceeds 1 MiB"}

        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(target, file_content)
        self._record_patch(name)
        self._build_index()
        return {"ok": True, "name": name, "file_path": file_path}

    def remove_supporting_file(self, name: str, file_path: str) -> dict:
        entry = self._index.get(name)
        if entry is None:
            return {"ok": False, "error": f"skill not found: {name}"}

        target = entry.path / file_path
        resolved = target.resolve()
        if not str(resolved).startswith(str(entry.path.resolve())):
            return {"ok": False, "error": "invalid file path"}
        if not target.exists():
            return {"ok": False, "error": "file not found"}

        target.unlink()
        self._record_patch(name)
        self._build_index()
        return {"ok": True, "name": name, "file_path": file_path}

    # ------------------------------------------------------------------
    # Usage / provenance
    # ------------------------------------------------------------------
    def record_use(self, name: str) -> None:
        if name not in self._usage:
            return
        now = _now()
        self._usage[name]["use_count"] = self._usage[name].get("use_count", 0) + 1
        self._usage[name]["last_used_at"] = now
        self._save_usage()

    def _record_view(self, name: str) -> None:
        if name not in self._usage:
            return
        now = _now()
        self._usage[name]["view_count"] = self._usage[name].get("view_count", 0) + 1
        self._usage[name]["last_viewed_at"] = now
        self._save_usage()

    def _record_patch(self, name: str) -> None:
        if name not in self._usage:
            return
        now = _now()
        self._usage[name]["patch_count"] = self._usage[name].get("patch_count", 0) + 1
        self._usage[name]["patch_generation"] = (
            self._usage[name].get("patch_generation", 0) + 1
        )
        self._usage[name]["last_patched_at"] = now
        self._save_usage()

    def _record_creation(self, name: str, category: str, created_by: str = "agent") -> None:
        now = _now()
        self._usage[name] = {
            "created_by": created_by,
            "created_at": now,
            "category": category,
            "use_count": 0,
            "view_count": 0,
            "patch_count": 0,
            "patch_generation": 1,
            "last_reused_patch_generation": 1,
            "last_used_at": None,
            "last_viewed_at": None,
            "last_patched_at": now,
            "state": "active",
            "pinned": False,
        }
        self._save_usage()

    def get_usage(self, name: Optional[str] = None) -> dict:
        if name:
            return self._usage.get(name, {})
        return dict(self._usage)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        self._index = {}
        # Global layer first, then project-local overrides.
        for layer_dir in (self.global_dir, self.project_dir):
            if not layer_dir.exists():
                continue
            for category_dir in sorted(layer_dir.iterdir()):
                if not category_dir.is_dir() or category_dir.name.startswith("."):
                    continue
                for skill_dir in sorted(category_dir.iterdir()):
                    if not skill_dir.is_dir():
                        continue
                    skill_md = skill_dir / "SKILL.md"
                    if not skill_md.exists():
                        continue
                    metadata, description = self._parse_skill_md(skill_md)
                    self._index[skill_dir.name] = SkillEntry(
                        name=skill_dir.name,
                        category=category_dir.name,
                        description=description,
                        path=skill_dir,
                        metadata=metadata,
                    )

    @staticmethod
    def _parse_skill_md(path: Path) -> Tuple[dict, str]:
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata: dict = {}
        description = ""

        # Simple YAML frontmatter parser.
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                try:
                    # Use safe minimal parsing; no PyYAML dependency required.
                    for line in frontmatter.splitlines():
                        if ":" in line:
                            k, v = line.split(":", 1)
                            metadata[k.strip()] = v.strip()
                except Exception:
                    pass
                text = parts[2]

        # Extract description from frontmatter if present, otherwise first
        # non-empty line after optional # title.
        if "description" in metadata:
            description = metadata["description"]
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                first = lines[0]
                if first.startswith("#"):
                    if len(lines) > 1:
                        description = lines[1]
                else:
                    description = first
        return metadata, description

    def _list_supporting_files(self, entry: SkillEntry) -> List[str]:
        files: List[str] = []
        for subdir in ALLOWED_SUBDIRS:
            path = entry.path / subdir
            if path.exists():
                for f in sorted(path.rglob("*")):
                    if f.is_file():
                        files.append(str(f.relative_to(entry.path)))
        return files

    @staticmethod
    def _is_valid_name(name: str) -> bool:
        return bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name))

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            shutil.move(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def _load_usage(self) -> None:
        if self.usage_path.exists():
            try:
                self._usage = json.loads(self.usage_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._usage = {}
        else:
            self._usage = {}

    def _save_usage(self) -> None:
        self.usage_path.write_text(
            json.dumps(self._usage, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _set_state(self, name: str, state: str) -> None:
        if name in self._usage:
            self._usage[name]["state"] = state
            self._save_usage()


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
