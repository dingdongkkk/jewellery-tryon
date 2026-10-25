"""SQLite storage shared by the API process and the worker process (WAL mode)."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,            -- earring | pendant | ring | bracelet
    original_path TEXT,
    cutout_path TEXT,
    mask_path TEXT,
    dimensions TEXT,                   -- json: cutout px size, approx real size
    reference_views TEXT,              -- json list of extra view paths
    anchors TEXT,                      -- json: attach point etc, normalised to cutout
    meta TEXT                          -- json: source, licence, supported flag
);
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    original_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tryon_jobs (
    id TEXT PRIMARY KEY,
    upload_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    status TEXT NOT NULL,              -- queued preparing generating compositing complete quota_exceeded failed
    placement TEXT NOT NULL,           -- json
    provider TEXT NOT NULL,
    seed INTEGER NOT NULL,
    settings TEXT NOT NULL,            -- json
    output_path TEXT,                  -- full try-on image; sibling files hold the other assets
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    timings TEXT                       -- json: seconds per stage
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON tryon_jobs(status, created_at);
"""

JSON_COLS = {"dimensions", "reference_views", "anchors", "meta", "placement", "settings", "timings"}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def db():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.executescript(SCHEMA)


def row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    for k in JSON_COLS & d.keys():
        if d[k] is not None:
            d[k] = json.loads(d[k])
    return d


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now() -> float:
    return time.time()


def dumps(v) -> str:
    return json.dumps(v)
