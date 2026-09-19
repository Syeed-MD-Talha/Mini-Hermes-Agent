"""
Hermes-style session_search tool.

Modes:
  - discovery: FTS5 keyword search across all sessions
  - scroll:    context around a known message id
  - read:      read a full session or head/tail slice
  - list:      list recent sessions

This tool makes no LLM calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .store import SessionStore


class SessionSearchTool:
    """On-demand episodic memory retrieval."""

    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "session_search",
                "description": (
                    "Search the agent's long-term conversation archive (episodic memory). "
                    "Use this when the user refers to something from a previous conversation "
                    "or when you need historical evidence. This tool makes no LLM calls."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["discovery", "scroll", "read", "list"],
                            "description": "Search mode.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Keywords for discovery mode.",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session id for scroll/read modes.",
                        },
                        "message_id": {
                            "type": "integer",
                            "description": "Message id to scroll around.",
                        },
                        "head": {
                            "type": "integer",
                            "description": "Read only first N messages.",
                        },
                        "tail": {
                            "type": "integer",
                            "description": "Read only last N messages.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results for discovery/list.",
                        },
                    },
                    "required": ["mode"],
                },
            },
        }

    def run(self, arguments: dict) -> dict:
        mode = arguments.get("mode")
        if mode == "discovery":
            query = arguments.get("query", "")
            limit = arguments.get("limit") or 5
            results = self.store.search(query, limit=limit)
            return {"ok": True, "mode": "discovery", "query": query, "results": results}

        if mode == "scroll":
            session_id = arguments.get("session_id")
            message_id = arguments.get("message_id")
            if not session_id or message_id is None:
                return {"ok": False, "error": "session_id and message_id required for scroll"}
            messages = self.store.scroll(session_id, int(message_id))
            return {"ok": True, "mode": "scroll", "messages": messages}

        if mode == "read":
            session_id = arguments.get("session_id")
            if not session_id:
                return {"ok": False, "error": "session_id required for read"}
            head = arguments.get("head")
            tail = arguments.get("tail")
            result = self.store.read_session(session_id, head=head, tail=tail)
            return {"ok": True, "mode": "read", **result}

        if mode == "list":
            limit = arguments.get("limit") or 10
            sessions = self.store.list_sessions(limit=limit)
            return {"ok": True, "mode": "list", "sessions": sessions}

        return {"ok": False, "error": f"unknown mode: {mode}"}
