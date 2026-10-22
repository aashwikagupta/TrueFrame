import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "trueframe.db"


def get_conn():
    conn = sqlite3.connect(os.environ.get("TRUEFRAME_DB", str(DEFAULT_DB)))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT UNIQUE NOT NULL,
                filename TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )


def save_prediction(image_hash, filename, prediction, confidence):
    with get_conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO predictions (image_hash, filename, prediction, confidence) VALUES (?, ?, ?, ?)",
            (image_hash, filename, prediction, confidence),
        )


def list_predictions(limit=50):
    with get_conn() as c:
        rows = c.execute(
            "SELECT image_hash, filename, prediction, confidence, created_at "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
