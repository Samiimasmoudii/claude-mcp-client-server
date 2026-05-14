"""
SQLite-backed conversation session manager.
Stores sessions at ~/.mcp-chat/sessions.db
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


_DB_PATH = Path.home() / ".mcp-chat" / "sessions.db"


class SessionManager:
    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    name        TEXT PRIMARY KEY,
                    messages    TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name TEXT,
                    role         TEXT,
                    content      TEXT,
                    timestamp    TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_history_session ON history(session_name);
                CREATE INDEX IF NOT EXISTS idx_history_content ON history(content);
            """)

    # ── sessions ──────────────────────────────────────────────────────────────

    def save(self, name: str, messages: list):
        now = datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute("SELECT created_at FROM sessions WHERE name=?", (name,)).fetchone()
            created_at = row[0] if row else now
            conn.execute(
                "INSERT OR REPLACE INTO sessions (name, messages, created_at, updated_at) VALUES (?,?,?,?)",
                (name, json.dumps(messages), created_at, now),
            )

    def load(self, name: str) -> Optional[list]:
        with self._connect() as conn:
            row = conn.execute("SELECT messages FROM sessions WHERE name=?", (name,)).fetchone()
            return json.loads(row[0]) if row else None

    def delete(self, name: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE name=?", (name,))

    def list_sessions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
            return [{"name": r[0], "updated_at": r[1]} for r in rows]

    # ── history ───────────────────────────────────────────────────────────────

    def log(self, session_name: str, role: str, content: str):
        # Only log plain text content
        if not isinstance(content, str):
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO history (session_name, role, content, timestamp) VALUES (?,?,?,?)",
                (session_name, role, content, datetime.now().isoformat(timespec="seconds")),
            )

    def search(self, query: str, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT session_name, role, content, timestamp
                   FROM history
                   WHERE content LIKE ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (f"%{query}%", limit),
            ).fetchall()
            return [
                {"session": r[0], "role": r[1], "content": r[2], "timestamp": r[3]}
                for r in rows
            ]
