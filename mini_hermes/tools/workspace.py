"""
Basic workspace tools: read_file, write_file, search_files, patch, execute_code.

These are the primitives that subagents (especially coder) need to actually
modify files and run commands in the workspace.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional


class WorkspaceTools:
    """File and shell primitives for the agent and subagents."""

    def __init__(self, workspace: str = ".") -> None:
        self.workspace = Path(workspace).resolve()

    def _safe_path(self, file_path: str) -> Path:
        target = (self.workspace / file_path).resolve()
        if not str(target).startswith(str(self.workspace)):
            raise ValueError("path escapes workspace")
        return target

    def read_file(self, file_path: str) -> dict:
        try:
            target = self._safe_path(file_path)
            if not target.exists():
                return {"ok": False, "error": f"file not found: {file_path}"}
            content = target.read_text(encoding="utf-8", errors="replace")
            return {"ok": True, "file_path": file_path, "content": content}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def write_file(self, file_path: str, content: str) -> dict:
        try:
            target = self._safe_path(file_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"ok": True, "file_path": file_path, "bytes": len(content.encode("utf-8"))}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def search_files(self, pattern: str, path: str = ".") -> dict:
        try:
            base = self._safe_path(path)
            matches: List[str] = []
            regex = re.compile(pattern)
            for root, _, files in os.walk(base):
                for name in files:
                    if name.endswith((".pyc", ".pyo")):
                        continue
                    full = Path(root) / name
                    try:
                        text = full.read_text(encoding="utf-8", errors="ignore")
                        if regex.search(text):
                            matches.append(str(full.relative_to(self.workspace)))
                    except Exception:
                        pass
            return {"ok": True, "pattern": pattern, "matches": matches}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def patch(self, file_path: str, old_string: str, new_string: str) -> dict:
        try:
            target = self._safe_path(file_path)
            if not target.exists():
                return {"ok": False, "error": f"file not found: {file_path}"}
            content = target.read_text(encoding="utf-8")
            if old_string not in content:
                return {"ok": False, "error": "old_string not found"}
            if content.count(old_string) > 1:
                return {"ok": False, "error": "old_string is ambiguous"}
            content = content.replace(old_string, new_string, 1)
            target.write_text(content, encoding="utf-8")
            return {"ok": True, "file_path": file_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def execute_code(self, command: str, cwd: Optional[str] = None) -> dict:
        """Execute a shell command and return stdout/stderr/exit code."""
        try:
            work_dir = self.workspace
            if cwd:
                work_dir = self._safe_path(cwd)
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "ok": result.returncode == 0,
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "command timed out"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def schemas(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write or overwrite a file in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search file contents with a regex pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "patch",
                    "description": "Replace a unique substring in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["file_path", "old_string", "new_string"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Run a shell command in the workspace and return output.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "cwd": {"type": "string"},
                        },
                        "required": ["command"],
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict) -> dict:
        if name == "read_file":
            return self.read_file(arguments.get("file_path", ""))
        if name == "write_file":
            return self.write_file(arguments.get("file_path", ""), arguments.get("content", ""))
        if name == "search_files":
            return self.search_files(arguments.get("pattern", ""), arguments.get("path", "."))
        if name == "patch":
            return self.patch(
                arguments.get("file_path", ""),
                arguments.get("old_string", ""),
                arguments.get("new_string", ""),
            )
        if name == "execute_code":
            return self.execute_code(arguments.get("command", ""), arguments.get("cwd"))
        return {"ok": False, "error": f"unknown workspace tool: {name}"}
