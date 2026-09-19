"""
Hermes-style episodic session storage.

Keeps all conversations in ~/.mini-hermes/state.db with FTS5 indexes
for fast keyword search.  Supports session lineage (parent sessions).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    parent_id TEXT REFERENCES sessions(id),
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    name TEXT,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content_rowid=rowid,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert
AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content)
    VALUES (NEW.rowid, COALESCE(NEW.content, '') || ' ' || COALESCE(NEW.tool_calls, ''));
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete
AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = OLD.rowid;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = OLD.rowid;
    INSERT INTO messages_fts(rowid, content)
    VALUES (NEW.rowid, COALESCE(NEW.content, '') || ' ' || COALESCE(NEW.tool_calls, ''));
END;

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
"""


@dataclass
class Message:
    id: Optional[int]
    session_id: str
    role: str
    content: Optional[str]
    tool_calls: Optional[str]
    tool_call_id: Optional[str]
    name: Optional[str]
    created_at: str


class SessionStore:
    """SQLite-backed session archive with FTS5 search."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path.home() / ".mini-hermes"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "state.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    def create_session(self, parent_id: Optional[str] = None, title: Optional[str] = None) -> str:
        now = _now()
        session_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute(
                "INSERT INTO sessions (id, parent_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, parent_id, title or "Untitled", now, now),
            )
            conn.commit()
        return session_id

    def update_session_title(self, session_id: str, title: str) -> None:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, _now(), session_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Message storage
    # ------------------------------------------------------------------
    def append_message(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[list] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> int:
        now = _now()
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, name, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, role, content, tool_calls_json, tool_call_id, name, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0,
        after_id: Optional[int] = None,
    ) -> List[Message]:
        sql = "SELECT * FROM messages WHERE session_id = ?"
        params: List = [session_id]
        if after_id is not None:
            sql += " AND id > ?"
            params.append(after_id)
        sql += " ORDER BY id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        if offset:
            sql += f" OFFSET {int(offset)}"
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_message(r) for r in rows]

    def get_message(self, message_id: int) -> Optional[Message]:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        return _row_to_message(row) if row else None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        limit: int = 10,
        context_messages: int = 2,
    ) -> List[dict]:
        """
        FTS5 discovery search. Returns matches grouped by session with
        local context around each hit.
        """
        if not query.strip():
            return []

        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            # Use bm25 for ranking.
            # Quote the query so FTS5 treats it as a literal phrase even if it
            # contains reserved words such as "hermes".
            safe_query = f'"{query.replace('"', '""')}"'
            rows = conn.execute(
                """
                SELECT m.id, m.session_id, m.role, m.content, m.tool_calls,
                       m.tool_call_id, m.name, m.created_at,
                       rank
                FROM messages_fts
                JOIN messages m ON messages_fts.rowid = m.rowid
                JOIN sessions s ON m.session_id = s.id
                WHERE messages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
            ).fetchall()

        results: List[dict] = []
        seen_sessions = set()
        for row in rows:
            session_id = row["session_id"]
            if session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            context = self._context_around(session_id, row["id"], context_messages)
            results.append(
                {
                    "session_id": session_id,
                    "match_message_id": row["id"],
                    "match_role": row["role"],
                    "match_preview": _preview(row["content"], 200),
                    "context": context,
                }
            )
        return results

    def scroll(
        self,
        session_id: str,
        around_message_id: int,
        before: int = 5,
        after: int = 5,
    ) -> List[dict]:
        """Return messages around a specific message id."""
        context = self._context_around(session_id, around_message_id, before, after)
        return context

    def read_session(
        self,
        session_id: str,
        head: Optional[int] = None,
        tail: Optional[int] = None,
    ) -> List[dict]:
        """Read a session, optionally only head or tail."""
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            count_row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
            total = count_row[0]

            if head is not None:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id LIMIT ?",
                    (session_id, head),
                ).fetchall()
            elif tail is not None:
                rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, tail),
                ).fetchall()
                rows = list(reversed(rows))
            else:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()

        return {
            "session_id": session_id,
            "total_messages": total,
            "messages": [_row_to_dict(r) for r in rows],
        }

    def list_sessions(self, limit: int = 20) -> List[dict]:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _context_around(
        self,
        session_id: str,
        message_id: int,
        before: int = 2,
        after: int = 2,
    ) -> List[dict]:
        with sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE session_id = ? AND id BETWEEN ? AND ?
                ORDER BY id
                """,
                (session_id, message_id - before, message_id + after),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        role=row["role"],
        content=row["content"],
        tool_calls=row["tool_calls"],
        tool_call_id=row["tool_call_id"],
        name=row["name"],
        created_at=row["created_at"],
    )


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "tool_calls": row["tool_calls"],
        "tool_call_id": row["tool_call_id"],
        "name": row["name"],
        "created_at": row["created_at"],
    }


def _preview(text: Optional[str], max_len: int) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
