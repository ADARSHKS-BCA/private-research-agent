"""
Lightweight SQLite Persistence for Multi-Turn Conversations and Chat History.

Manages conversation sessions, message history, retrieved sources, and thinking steps.
Zero external database service required.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

from app.config import settings

logger = logging.getLogger(__name__)


class ConversationStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or settings.sqlite_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    steps_json TEXT,
                    sources_json TEXT,
                    timestamp INTEGER NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
            conn.commit()

    def create_conversation(self, title: str = "", conversation_id: Optional[str] = None) -> str:
        cid = conversation_id or str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc).isoformat()
        conv_title = title.strip() or f"Research Session {cid}"

        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (cid, conv_title, now, now),
            )
            conn.commit()
        return cid

    def list_conversations(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            conv = dict(row)
            conv["messages"] = self.get_messages(conversation_id)
            return conv

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()
            return cur.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        # Ensure conversation exists
        self.create_conversation(
            title=content[:60] if role == "user" else "",
            conversation_id=conversation_id,
        )

        mid = str(uuid.uuid4())
        now_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        now_iso = datetime.now(timezone.utc).isoformat()
        steps_str = json.dumps(steps) if steps else None
        sources_str = json.dumps(sources) if sources else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, steps_json, sources_json, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (mid, conversation_id, role, content, steps_str, sources_str, now_ts),
            )
            # Update conversation timestamp and title if first user message
            if role == "user":
                conn.execute(
                    "UPDATE conversations SET updated_at = ?, title = COALESCE(NULLIF(title, ''), ?) WHERE id = ?",
                    (now_iso, content[:60], conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now_iso, conversation_id),
                )
            conn.commit()
        return mid

    def get_messages(self, conversation_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT id, role, content, steps_json, sources_json, timestamp
                FROM messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = cur.fetchall()
            messages = []
            for r in rows:
                m = {
                    "id": r["id"],
                    "role": r["role"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "steps": json.loads(r["steps_json"]) if r["steps_json"] else [],
                    "sources": json.loads(r["sources_json"]) if r["sources_json"] else [],
                }
                messages.append(m)
            return messages

    def get_recent_chat_history(
        self,
        conversation_id: str,
        window: int = 6,
    ) -> List[Dict[str, str]]:
        """
        Return last N messages formatted as [{'role': 'user', 'content': '...'}, ...]
        for LLM prompt context conditioning.
        """
        if not conversation_id:
            return []

        msgs = self.get_messages(conversation_id, limit=window)
        return [{"role": m["role"], "content": m["content"]} for m in msgs]


# Global conversation store singleton
_conv_store = None


def get_conversation_store() -> ConversationStore:
    global _conv_store
    if _conv_store is None:
        _conv_store = ConversationStore()
    return _conv_store
