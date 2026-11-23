"""Read the whole Luma graph as plain data for visualisation.

Opens its own READ-ONLY connection, so it works while the CLI / server holds the writer, and it can
never modify anything. The result is JSON-friendly: no Relate objects, no datetimes, no embeddings.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import relate

BASE_LABELS = ("Entity", "Event")
MEDIA_LABELS = frozenset({"Session", "Image", "AudioClip", "TextBlock"})
PERCEPTION_LABELS = frozenset({"Detection", "Utterance"})
MAX_TEXT = 600           # long values (OCR text, notes) are cut for the browser
LABEL_CHARS = 28         # characters of a node's name shown on the canvas
# When a graph is too big, keep layers in this order of importance.
LAYER_PRIORITY = ("entity", "event", "other", "perception", "media", "registry")


def classify(labels: list[str]) -> tuple[str, str]:
    """(kind, layer) for a node. kind = its most specific label; layer = where it sits in Luma's design."""
    kind = next((l for l in labels if l not in BASE_LABELS), labels[0] if labels else "Unlabeled")
    names = set(labels)
    if "EntityType" in names:
        return kind, "registry"
    if names & MEDIA_LABELS:
        return kind, "media"
    if names & PERCEPTION_LABELS:
        return kind, "perception"
    if "Event" in names:
        return kind, "event"
    if "Entity" in names:
        return kind, "entity"
    return kind, "other"


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def display_name(kind: str, props: dict) -> str:
    for key in ("name", "title", "label", "text", "key", "id"):
        if props.get(key):
            return _short(props[key], LABEL_CHARS)
    return kind


def clean_value(key: str, value: Any) -> Any:
    if key == "embedding" and isinstance(value, (list, tuple)):
        return f"[{len(value)} numbers, hidden]"
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [clean_value("", v) for v in value[:50]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _short(value, MAX_TEXT) if isinstance(value, str) else value
    return _short(str(value), MAX_TEXT)  # anything unusual (vectors, points, ...) becomes text


def clean_props(props: dict | None) -> dict:
    return {k: clean_value(k, v) for k, v in (props or {}).items()}


def _rows(conn, query: str) -> list[dict]:
    cur = conn.cursor()
    cur.execute(query)
    return cur.fetchall()


def read_graph(db_path: str, max_nodes: int = 3000) -> dict:
    """Snapshot of the database: {"nodes": [...], "edges": [...], "stats": {...}}."""
    conn = relate.connect(db_path, read_only=True)
    try:
        node_rows = _rows(conn, "MATCH (n) RETURN id(n) AS nid, labels(n) AS labels, properties(n) AS props")
        edge_rows = _rows(conn, "MATCH (a)-[r]->(b) RETURN id(r) AS eid, id(a) AS s, id(b) AS t, "
                                "type(r) AS type, properties(r) AS props")
    finally:
        conn.close()

    nodes = []
    for r in node_rows:
        labels = list(r["labels"])
        kind, layer = classify(labels)
        props = clean_props(r["props"])
        nodes.append({"id": r["nid"], "labels": labels, "kind": kind, "layer": layer,
                      "name": display_name(kind, props), "props": props})

    truncated = len(nodes) > max_nodes
    if truncated:  # keep the most meaningful layers first, then by id for stability
        rank = {layer: i for i, layer in enumerate(LAYER_PRIORITY)}
        nodes = sorted(nodes, key=lambda n: (rank.get(n["layer"], 99), n["id"]))[:max_nodes]
    keep = {n["id"] for n in nodes}
    edges = [{"id": r["eid"], "from": r["s"], "to": r["t"], "type": r["type"], "props": clean_props(r["props"])}
             for r in edge_rows if r["s"] in keep and r["t"] in keep]

    return {"nodes": sorted(nodes, key=lambda n: n["id"]), "edges": edges,
            "stats": {"nodes": len(nodes), "edges": len(edges), "total_nodes": len(node_rows),
                      "truncated": truncated,
                      "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}}