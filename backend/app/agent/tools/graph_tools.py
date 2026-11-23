"""Generic graph tools: schema discovery, creating/updating/linking nodes, searching, raw reads.

These give the agent its freedom. Every label and relationship type must be registered
(app.db.registry) before use, and the media/perception layers are off limits.
"""
import re
from typing import Any, Literal

from agents import RunContextWrapper

from backend.app.agent.context import LumaContext
from backend.app.agent.tools import base
from backend.app.agent.tools.base import ToolError, clean_props, luma_tool, new_id
from backend.app.db import registry
from backend.app.db.client import LumaDB
from backend.app.db.guards import (assert_agent_may_use_label, normalize_label, normalize_rel_type,
                           validate_label, validate_rel_type)

RESERVED_ON_CREATE = frozenset({"id", "name", "created_at", "updated_at"})
RESERVED_ON_UPDATE = frozenset({"id", "created_at", "updated_at"})


# ---- helpers (plain functions that take the db; reused by memory_tools) -------------------
def public(node: dict | None) -> dict | None:
    """Drop embeddings: huge, and useless to an LLM."""
    return None if node is None else {k: v for k, v in node.items() if k != "embedding"}


def strip_embeddings(value):
    if isinstance(value, dict):
        return {k: strip_embeddings(v) for k, v in value.items() if k != "embedding"}
    if isinstance(value, list):
        return [strip_embeddings(v) for v in value]
    return value


def node_base(db: LumaDB, node_id: str) -> str | None:
    """'Entity' or 'Event' for the node with this id, or None. Agent tools only touch these two."""
    for b in ("Entity", "Event"):
        if db.read(f"MATCH (n:{b} {{id:$id}}) RETURN count(n) AS c", {"id": node_id})[0]["c"]:
            return b
    return None


def require_node(db: LumaDB, node_id: str) -> str:
    b = node_base(db, node_id)
    if b is None:
        raise ToolError(f"No node with id {node_id!r}. Use find_entities to look up ids.")
    return b


def find_by_name(db: LumaDB, label: str | None, name: str) -> dict | None:
    """Case-insensitive match on name or alias."""
    lab = f":{validate_label(label)}" if label else ""
    rows = db.read(
        f"MATCH (n:Entity{lab}) WHERE toLower(n.name)=$q OR $q IN [x IN coalesce(n.aliases, []) | toLower(x)] "
        "RETURN n LIMIT 1", {"q": name.strip().lower()})
    return rows[0]["n"] if rows else None


def insert_node(db: LumaDB, label: str, base_label: str, name: str, props: dict) -> dict:
    """Pure write (no reads), so it is safe inside db.transaction()."""
    now = base.utcnow()
    record = {**props, "id": new_id(label.lower()[:8]), "name": name, "created_at": now, "updated_at": now}
    rows = db.write(f"CREATE (n:{base_label}:{validate_label(label)} $props) RETURN n", {"props": record})
    return public(rows[0]["n"])


def link_known(db: LumaDB, from_id: str, from_base: str, rel: str, to_id: str, to_base: str,
               props: dict | None = None) -> None:
    """Pure write. MERGE makes repeating the same link harmless."""
    q = (f"MATCH (a:{from_base} {{id:$f}}), (b:{to_base} {{id:$t}}) "
         f"MERGE (a)-[r:{validate_rel_type(rel)}]->(b)")
    if props:
        q += " SET r += $props"
    db.write(q, {"f": from_id, "t": to_id, **({"props": props} if props else {})})


def ensure_entity(db: LumaDB, label: str, name: str, props: dict | None = None) -> dict:
    found = find_by_name(db, label, name)
    return public(found) if found else insert_node(db, label, "Entity", name, props or {})


# ---- schema tools -------------------------------------------------------------------------
@luma_tool
def list_types(ctx: RunContextWrapper[LumaContext], kind: Literal["node", "relationship"] | None = None) -> dict:
    """List the node labels and relationship types that exist. ALWAYS check this before creating
    anything so you reuse existing types instead of inventing near-duplicates.

    Args:
        kind: Only node labels or only relationship types. Omit for both.
    """
    keep = ("name", "kind", "base", "description", "example_props", "aliases")
    return {"types": [{k: t[k] for k in keep if t.get(k) not in (None, [])}
                      for t in registry.list_types(ctx.context.db, kind)
                      if t["layer"] not in ("media", "perception")]}


@luma_tool
def define_type(ctx: RunContextWrapper[LumaContext], name: str, kind: Literal["node", "relationship"],
                description: str, base: Literal["Entity", "Event"] = "Entity",
                example_props: list[str] | None = None, aliases: list[str] | None = None,
                force: bool = False) -> dict:
    """Register a NEW node label or relationship type when nothing in list_types fits. If a similar
    type already exists you get it back instead of a new one, so reuse it.

    Args:
        name: The new type's name, e.g. 'Vehicle' or 'ALLERGIC_TO'. It is normalised for you.
        kind: 'node' for a label, 'relationship' for a connection type.
        description: One line on what it represents.
        base: For nodes: 'Entity' for persistent things (a bike, a shop), 'Event' for things that
            happen at a time (a meal, a visit).
        example_props: Typical property names, e.g. ['plate', 'colour'].
        aliases: Other names for the same idea, e.g. ['car'].
        force: Create even though a similar type exists. Rarely the right choice.
    """
    return registry.define_type(ctx.context.db, name, kind, description, example_props, aliases, force, base)


# ---- write tools --------------------------------------------------------------------------
@luma_tool(strict=False)
def create_node(ctx: RunContextWrapper[LumaContext], label: str, name: str,
                properties: dict[str, Any] | None = None, allow_duplicate: bool = False) -> dict:
    """Create a node of a registered type. For things, an existing node with the same name is
    returned instead of making a duplicate. People, sightings, appointments and reminders have
    dedicated tools; use those when they fit.

    Args:
        label: A type from list_types, e.g. Object, Place, Vehicle.
        name: Human name, e.g. 'blue umbrella'.
        properties: Flat key/values: text, numbers, booleans, or lists of those. No nested objects.
        allow_duplicate: Create even if a node with this name already exists.
    """
    db = ctx.context.db
    label = normalize_label(label)
    assert_agent_may_use_label(label)
    t = registry.get_type(db, label, "node")
    if t is None:
        raise ToolError(f"Node type {label!r} is not registered. Call list_types and reuse a similar "
                        f"type, or define_type first.")
    props = clean_props(properties, RESERVED_ON_CREATE)
    if t["base"] == "Entity" and not allow_duplicate:
        existing = find_by_name(db, label, name)
        if existing:
            return {"created": False, "node": public(existing),
                    "hint": "Already exists; use this id (update_node to change it)."}
    return {"created": True, "node": insert_node(db, label, t["base"], name, props)}


@luma_tool(strict=False)
def update_node(ctx: RunContextWrapper[LumaContext], id: str, properties: dict[str, Any]) -> dict:
    """Change properties on an existing node. Properties set to null are ignored.

    Args:
        id: The node's id (from find_entities or an earlier result).
        properties: Flat key/values to set. Existing values with the same key are replaced.
    """
    db = ctx.context.db
    b = require_node(db, id)
    props = clean_props(properties, RESERVED_ON_UPDATE)
    rows = db.write(f"MATCH (n:{b} {{id:$id}}) SET n += $props, n.updated_at=$now RETURN n",
                    {"id": id, "props": props, "now": base.utcnow()})
    return {"node": public(rows[0]["n"])}


@luma_tool(strict=False)
def create_relationship(ctx: RunContextWrapper[LumaContext], from_id: str, rel_type: str, to_id: str,
                        properties: dict[str, Any] | None = None) -> dict:
    """Connect two existing nodes with a registered relationship type, e.g. OWNED_BY or KNOWS.
    Repeating the same link is harmless.

    Args:
        from_id: Id of the source node.
        rel_type: A relationship type from list_types. Normalised for you ('owned by' -> OWNED_BY).
        to_id: Id of the target node.
        properties: Optional flat key/values stored on the relationship.
    """
    db = ctx.context.db
    rel = normalize_rel_type(rel_type)
    t = registry.get_type(db, rel, "relationship")
    if t is None:
        raise ToolError(f"Relationship type {rel!r} is not registered. Call list_types, or define_type "
                        f"(kind='relationship') first.")
    if t["layer"] in ("media", "perception"):
        raise ToolError(f"{rel} is managed by Luma's perception pipeline; you cannot create it.")
    props = clean_props(properties, frozenset())
    link_known(db, from_id, require_node(db, from_id), rel, to_id, require_node(db, to_id), props)
    return {"linked": f"{from_id} -[{rel}]-> {to_id}"}


# ---- read tools ---------------------------------------------------------------------------
@luma_tool
def find_entities(ctx: RunContextWrapper[LumaContext], query: str, label: str | None = None,
                  limit: int = 10) -> dict:
    """Search persistent things (people, objects, places, ...) by name, alias or text. Exact name
    or alias matches come first. Use this to get ids before linking or updating.

    Args:
        query: What to look for, e.g. 'umbrella' or 'Dr Karim'.
        label: Optional type filter, e.g. 'Person'.
        limit: Maximum results (1-25).
    """
    db = ctx.context.db
    limit = max(1, min(limit, 25))
    lab = f":{normalize_label(label)}" if label else ""
    q = query.strip()
    results = db.read(
        f"MATCH (n:Entity{lab}) WHERE toLower(n.name)=$q OR $q IN [x IN coalesce(n.aliases, []) | toLower(x)] "
        "RETURN n LIMIT $limit", {"q": q.lower(), "limit": limit})
    matches = [{"match": "exact", "node": public(r["n"])} for r in results]
    tokens = re.findall(r"[A-Za-z0-9]+", q)
    if tokens and len(matches) < limit:
        seen = {m["node"]["id"] for m in matches}
        where = f" WHERE node:{normalize_label(label)}" if label else ""
        for r in db.read(f"CALL db.index.fulltext.queryNodes('entity_text', $q) YIELD node, score{where} "
                         "RETURN node, score ORDER BY score DESC LIMIT $limit",
                         {"q": " OR ".join(tokens), "limit": limit}):
            if r["node"]["id"] not in seen and len(matches) < limit:
                matches.append({"match": "text", "node": public(r["node"])})
    return {"matches": matches}


@luma_tool
def get_node(ctx: RunContextWrapper[LumaContext], id: str) -> dict:
    """Show one node with its direct connections (what it links to and what links to it).

    Args:
        id: The node's id.
    """
    db = ctx.context.db
    b = require_node(db, id)
    node = db.read(f"MATCH (n:{b} {{id:$id}}) RETURN n", {"id": id})[0]["n"]
    out = db.read(f"MATCH (n:{b} {{id:$id}})-[r]->(m) RETURN r, m LIMIT 25", {"id": id})
    inc = db.read(f"MATCH (n:{b} {{id:$id}})<-[r]-(m) RETURN r, m LIMIT 25", {"id": id})
    brief = lambda m: {k: m[k] for k in ("id", "name", "_labels") if k in m}  # noqa: E731
    return {"node": public(node),
            "connections": [{"direction": "out", "rel": r["r"]["_type"], "node": brief(r["m"])} for r in out] +
                           [{"direction": "in", "rel": r["r"]["_type"], "node": brief(r["m"])} for r in inc]}


@luma_tool(strict=False)
def run_read_query(ctx: RunContextWrapper[LumaContext], cypher: str, params: dict[str, Any] | None = None,
                   max_rows: int = 25) -> dict:
    """Run a read-only Cypher query for questions the other tools can't answer. Writes are
    impossible. Pass values via $params, and add your own ORDER BY / LIMIT.

    Args:
        cypher: The Cypher query.
        params: Values referenced as $name in the query.
        max_rows: Row cap (1-100). The result says if it was truncated.
    """
    out = ctx.context.db.read_capped(cypher, params or {}, max_rows=max(1, min(max_rows, 100)))
    out["rows"] = strip_embeddings(out["rows"])
    return out