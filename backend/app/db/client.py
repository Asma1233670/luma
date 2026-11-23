"""LumaDB: the only module that opens the Relate database.

Constraints discovered against relate-db 1.0.1:
  * A database allows ONE writer connection ("storage locked" otherwise).
  * Connections are THREAD-AFFINE: a connection may only be used by the thread that opened it.
  * A read-only connection is a snapshot taken when it opens; a fresh one sees all earlier writes.

Design:
  * One dedicated "owner" thread opens and uses the writer connection. Every write is shipped to
    it, so LumaDB can be called from ANY thread (agent tool workers, web request threads,
    schedulers) without tripping thread affinity.
  * A lock serialises callers, and makes transaction() atomic with respect to other callers.
  * Reads flush the writer, then open a fresh READ-ONLY connection in the calling thread, so
    agent-authored Cypher physically cannot modify the graph.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Iterator

import relate


def to_plain(value: Any) -> Any:
    """Convert Relate Node/Relationship objects into JSON-friendly dicts."""
    if isinstance(value, relate.Node):
        return {"_labels": list(value.labels), **dict(value.properties)}
    if isinstance(value, relate.Relationship):
        return {"_type": value.type, **dict(value.properties)}
    if isinstance(value, list):
        return [to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    return value


class LumaDB:
    def __init__(self, path: str):
        if path == ":memory:":
            raise ValueError("LumaDB needs a file-backed database (read-only snapshots can't see :memory:).")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # Relate creates `path` itself
        self.path = path
        self._lock = threading.RLock()
        self._owner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="luma-db")
        self._writer = self._on_owner(relate.connect, path)  # opened BY the owner thread
        self._closed = False

    def _on_owner(self, fn, *args):
        return self._owner.submit(fn, *args).result()

    # ---- writes -------------------------------------------------------------
    def write(self, query: str, params: dict | None = None) -> list[dict]:
        with self._lock:
            return self._on_owner(self._run, self._writer, query, params)

    @contextmanager
    def transaction(self) -> Iterator["LumaDB"]:
        """All-or-nothing group of writes: `with db.transaction(): db.write(...)`.
        Don't read inside it: read-only snapshots can't see uncommitted writes."""
        with self._lock:
            self._on_owner(setattr, self._writer, "autocommit", False)
            try:
                yield self
                self._on_owner(self._writer.commit)
            except BaseException:
                self._on_owner(self._writer.rollback)
                raise
            finally:
                self._on_owner(setattr, self._writer, "autocommit", True)

    # ---- reads --------------------------------------------------------------
    def read(self, query: str, params: dict | None = None) -> list[dict]:
        """Run a query on a fresh read-only connection (cannot write by construction)."""
        with self._lock:
            self._on_owner(self._writer.flush)
        ro = relate.connect(self.path, read_only=True)  # opened and used in THIS thread
        try:
            return self._run(ro, query, params)
        finally:
            ro.close()

    def read_capped(self, query: str, params: dict | None = None, max_rows: int = 50) -> dict:
        """Like read(), but bounded so an agent can't flood its own context window."""
        rows = self.read(query, params)
        return {"rows": rows[:max_rows], "truncated": len(rows) > max_rows, "total": len(rows)}

    # ---- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._on_owner(self._writer.flush)
            self._on_owner(self._writer.close)
            self._owner.shutdown(wait=True)

    @staticmethod
    def _run(conn, query: str, params: dict | None) -> list[dict]:
        cur = conn.cursor()
        cur.execute(query, params or {})
        try:
            rows = cur.fetchall()
        except Exception:  # statements with no result set (CREATE INDEX, etc.)
            return []
        return [{k: to_plain(v) for k, v in row.items()} for row in rows]