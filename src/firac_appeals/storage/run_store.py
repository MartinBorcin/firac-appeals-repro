from __future__ import annotations

import json
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any

from firac_appeals.utils.io import ensure_parent

COMPRESS_THRESHOLD_BYTES = 8192


class RunStore:
    """Compact SQLite-backed storage for per-case run artifacts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_parent(self.path)
        self.connection = sqlite3.connect(str(self.path), timeout=60, isolation_level=None)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS items (
                item_type TEXT NOT NULL,
                case_id TEXT NOT NULL,
                name TEXT NOT NULL,
                payload_json TEXT,
                payload_blob BLOB,
                payload_encoding TEXT,
                text TEXT,
                text_blob BLOB,
                text_encoding TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (item_type, case_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_items_case ON items(case_id);
            CREATE INDEX IF NOT EXISTS idx_items_type_name ON items(item_type, name);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                case_id TEXT NOT NULL,
                name TEXT NOT NULL,
                payload_json TEXT,
                text TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_case ON events(case_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, name);
            """
        )
        self._ensure_column("items", "payload_blob", "BLOB")
        self._ensure_column("items", "payload_encoding", "TEXT")
        self._ensure_column("items", "text_blob", "BLOB")
        self._ensure_column("items", "text_encoding", "TEXT")

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        existing = {
            row[1]
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def close(self) -> None:
        self.connection.close()

    def put_json(self, item_type: str, case_id: str, name: str, payload: dict[str, Any]) -> None:
        payload_json, payload_blob, payload_encoding = encode_maybe_compressed(
            json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        self.put(
            item_type,
            case_id,
            name,
            payload_json=payload_json,
            payload_blob=payload_blob,
            payload_encoding=payload_encoding,
        )

    def get_json(self, item_type: str, case_id: str, name: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json, payload_blob, payload_encoding FROM items WHERE item_type=? AND case_id=? AND name=?",
            (item_type, case_id, name),
        ).fetchone()
        if not row:
            return None
        text = decode_maybe_compressed(row[0], row[1], row[2])
        return json.loads(text) if text else None

    def put_text(self, item_type: str, case_id: str, name: str, text: str) -> None:
        raw_text, text_blob, text_encoding = encode_maybe_compressed(text)
        self.put(item_type, case_id, name, text=raw_text, text_blob=text_blob, text_encoding=text_encoding)

    def get_text(self, item_type: str, case_id: str, name: str) -> str | None:
        row = self.connection.execute(
            "SELECT text, text_blob, text_encoding FROM items WHERE item_type=? AND case_id=? AND name=?",
            (item_type, case_id, name),
        ).fetchone()
        return None if not row else decode_maybe_compressed(row[0], row[1], row[2])

    def put(
        self,
        item_type: str,
        case_id: str,
        name: str,
        payload_json: str | None = None,
        text: str | None = None,
        payload_blob: bytes | None = None,
        payload_encoding: str | None = None,
        text_blob: bytes | None = None,
        text_encoding: str | None = None,
    ) -> None:
        now = time.time()
        self.connection.execute(
            """
            INSERT INTO items(
                item_type, case_id, name, payload_json, payload_blob, payload_encoding,
                text, text_blob, text_encoding, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_type, case_id, name) DO UPDATE SET
                payload_json=excluded.payload_json,
                payload_blob=excluded.payload_blob,
                payload_encoding=excluded.payload_encoding,
                text=excluded.text,
                text_blob=excluded.text_blob,
                text_encoding=excluded.text_encoding,
                updated_at=excluded.updated_at
            """,
            (
                item_type,
                case_id,
                name,
                payload_json,
                payload_blob,
                payload_encoding,
                text,
                text_blob,
                text_encoding,
                now,
                now,
            ),
        )

    def append_event(
        self,
        event_type: str,
        case_id: str,
        name: str,
        payload: dict[str, Any] | None = None,
        text: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO events(event_type, case_id, name, payload_json, text, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                case_id,
                name,
                json.dumps(payload, ensure_ascii=False, sort_keys=True) if payload is not None else None,
                text,
                time.time(),
            ),
        )

    def has_json(self, item_type: str, case_id: str, name: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM items
            WHERE item_type=? AND case_id=? AND name=?
              AND (payload_json IS NOT NULL OR payload_blob IS NOT NULL)
            """,
            (item_type, case_id, name),
        ).fetchone()
        return bool(row)

    def count(self, item_type: str) -> int:
        row = self.connection.execute("SELECT COUNT(*) FROM items WHERE item_type=?", (item_type,)).fetchone()
        return int(row[0] if row else 0)

    def iter_items(self, item_type: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT case_id, name, payload_json, payload_blob, payload_encoding,
                   text, text_blob, text_encoding, updated_at
            FROM items WHERE item_type=? ORDER BY case_id, name
            """,
            (item_type,),
        ).fetchall()
        return [
            {
                "case_id": row[0],
                "name": row[1],
                "payload": json.loads(decode_maybe_compressed(row[2], row[3], row[4]))
                if decode_maybe_compressed(row[2], row[3], row[4])
                else None,
                "text": decode_maybe_compressed(row[5], row[6], row[7]),
                "updated_at": row[8],
            }
            for row in rows
        ]


def encode_maybe_compressed(text: str) -> tuple[str | None, bytes | None, str | None]:
    data = text.encode("utf-8")
    if len(data) < COMPRESS_THRESHOLD_BYTES:
        return text, None, None
    return None, zlib.compress(data, level=6), "zlib"


def decode_maybe_compressed(text: str | None, blob: bytes | None, encoding: str | None) -> str:
    if text is not None:
        return text
    if blob is None:
        return ""
    if encoding == "zlib":
        return zlib.decompress(blob).decode("utf-8")
    return bytes(blob).decode("utf-8")


def store_from_output_config(output_dir: str | Path, output_config: dict[str, Any]) -> RunStore | None:
    mode = str(output_config.get("artifact_store", "sqlite")).lower()
    if mode in {"", "files", "file", "none", "disabled"}:
        return None
    if mode != "sqlite":
        raise ValueError(f"Unsupported artifact_store: {mode}")
    filename = str(output_config.get("artifact_store_filename", "artifacts.sqlite"))
    return RunStore(Path(output_dir) / filename)
