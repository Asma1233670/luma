"""The schema registry: Luma's self-describing, agent-extensible type system.

Every node label and relationship type is an :EntityType node. Built-ins ship with
Luma; the agent can add more via define_type(), but must reuse what exists, which 
for example keeps 'Glasses' / 'Eyeglasses' / 'Spectacles' from becoming three different types.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher

from .client import LumaDB
from .guards import (assert_agent_may_use_label, normalize_label, normalize_rel_type)

SIMILARITY_THRESHOLD = 0.8

# (name, kind, layer, description, example_props)
BUILTIN_TYPES: list[tuple[str, str, str, str, list[str]]] = [
    # media layer
    ("Session", "node", "media", "One continuous use of Luma; groups captures", ["started_at", "ended_at"]),
    ("Image", "node", "media", "A camera frame Luma looked at", ["uri", "captured_at", "caption", "embedding"]),
    ("AudioClip", "node", "media", "A piece of audio Luma heard", ["uri", "captured_at", "transcript", "embedding"]),
    ("TextBlock", "node", "media", "Text from OCR or typed/pasted input", ["text", "origin", "embedding"]),
    # perception layer
    ("Detection", "node", "perception", "A region of an Image the vision model recognised", ["label", "bbox", "confidence", "embedding"]),
    ("Utterance", "node", "perception", "A segment of speech inside an AudioClip", ["text", "speaker_hint", "start_ms", "end_ms"]),
    # world model: persistent things
    ("Person", "node", "world", "A person the user knows or met", ["name", "phone", "email", "notes"]),
    ("Object", "node", "world", "A physical item the user owns or cares about", ["name", "category", "description"]),
    ("Place", "node", "world", "A location: room, building, shop", ["name", "kind", "details"]),
    ("Document", "node", "world", "A paper, letter, label or other readable item", ["title", "kind", "content"]),
    ("Medication", "node", "world", "A medicine, as read from a label or told by a person", ["name", "dosage", "instructions", "warnings"]),
    # world model: durable events
    ("Sighting", "node", "world", "Durable record that something was perceived somewhere", ["at", "confidence", "context"]),
    ("Memory", "node", "world", "Durable record of something said or read", ["at", "text", "kind"]),
    ("Appointment", "node", "world", "A scheduled meeting or visit", ["title", "start", "end", "status"]),
    ("Reminder", "node", "world", "Something to remind the user of at a given time", ["text", "due", "status"]),
    # relationships
    ("CONTAINS", "relationship", "perception", "Media contains a perceived item", []),
    ("HAS_TEXT", "relationship", "perception", "Media or document has extracted text", []),
    ("NEXT", "relationship", "media", "Temporal order between captures", []),
    ("IN_SESSION", "relationship", "media", "Capture belongs to a session", []),
    ("NEAR", "relationship", "perception", "Spatial: close to (within a frame)", []),
    ("ON", "relationship", "perception", "Spatial: resting on", []),
    ("INSIDE", "relationship", "perception", "Spatial: inside of", []),
    ("LEFT_OF", "relationship", "perception", "Spatial: to the left of", []),
    ("SIGHTING_OF", "relationship", "perception", "Detection resolved to a persistent Entity", ["confidence"]),
    ("PROMOTED_TO", "relationship", "perception", "Detection was consolidated into a Sighting", []),
    ("TAKEN_IN", "relationship", "world", "Image was captured at a Place", []),
    ("OF", "relationship", "world", "Sighting/Memory is about an Entity", []),
    ("IN", "relationship", "world", "Sighting happened in a Place", []),
    ("WITH", "relationship", "world", "Present together / participant", ["role"]),
    ("EVIDENCE", "relationship", "world", "Sighting is backed by an Image (while it exists)", []),
    ("MENTIONS", "relationship", "world", "Text or speech mentions an Entity", []),
    ("SAID_BY", "relationship", "world", "Memory was said by a Person", []),
    ("KNOWS", "relationship", "world", "Person knows Person", ["relationship", "since"]),
    ("OWNED_BY", "relationship", "world", "Object belongs to a Person", []),
    ("AT", "relationship", "world", "Event takes place at a Place", []),
    ("ABOUT", "relationship", "world", "Reminder/Memory concerns an Entity or Event", []),
    ("DESCRIBES", "relationship", "world", "Document describes something", []),
]


# Durable events carry the :Event base label; every other node type is an :Entity.
EVENT_TYPES = frozenset({"Sighting", "Memory", "Appointment", "Reminder"})


def _key(kind: str, name: str) -> str:
    return f"{kind}:{name.lower()}"


def seed_builtin_types(db: LumaDB) -> None:
    now = datetime.now(timezone.utc)
    for name, kind, layer, desc, props in BUILTIN_TYPES:
        db.write(
            "MERGE (t:EntityType {key:$key}) "
            "ON CREATE SET t.name=$name, t.kind=$kind, t.layer=$layer, t.description=$desc, "
            "t.example_props=$props, t.aliases=[], t.builtin=true, t.created_at=$now "
            "SET t.base=$base",
            {"key": _key(kind, name), "name": name, "kind": kind, "layer": layer,
             "desc": desc, "props": props, "now": now,
             "base": ("Event" if name in EVENT_TYPES else "Entity") if kind == "node" else None},
        )


def list_types(db: LumaDB, kind: str | None = None) -> list[dict]:
    where = "WHERE t.kind = $kind " if kind else ""
    return db.read(
        f"MATCH (t:EntityType) {where}"
        "RETURN t.name AS name, t.kind AS kind, t.layer AS layer, t.description AS description, "
        "t.example_props AS example_props, t.aliases AS aliases, t.builtin AS builtin, t.base AS base "
        "ORDER BY t.kind, t.name",
        {"kind": kind} if kind else {},
    )


def _normalize(kind: str, name: str) -> str:
    return normalize_label(name) if kind == "node" else normalize_rel_type(name)


def _flat(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_similar(db: LumaDB, name: str, kind: str, limit: int = 5) -> list[dict]:
    """Existing types that look like `name`, by exact/alias match or fuzzy name similarity."""
    target = _flat(name)
    scored = []
    for t in list_types(db, kind):
        names = [t["name"], *(t["aliases"] or [])]
        best = max(SequenceMatcher(None, target, _flat(n)).ratio() for n in names)
        if best >= SIMILARITY_THRESHOLD:
            scored.append({**t, "similarity": round(best, 2)})
    return sorted(scored, key=lambda t: -t["similarity"])[:limit]


def define_type(db: LumaDB, name: str, kind: str, description: str,
                example_props: list[str] | None = None, aliases: list[str] | None = None,
                force: bool = False, base: str = "Entity") -> dict:
    """Register a new node label or relationship type, or return the existing match.

    Returns {"created": bool, "type": {...}, "similar": [...], "hint": str}.
    If something similar already exists the call does NOT create, so the agent reuses it.
    Pass force=True only when the agent is sure the new type is genuinely distinct.
    """
    if kind not in ("node", "relationship"):
        raise ValueError("kind must be 'node' or 'relationship'")
    if base not in ("Entity", "Event"):
        raise ValueError("base must be 'Entity' (a persistent thing) or 'Event' (something that happened/will happen)")
    canonical = _normalize(kind, name)
    if kind == "node":
        assert_agent_may_use_label(canonical)

    existing = find_similar(db, canonical, kind)
    exact = [t for t in existing if t["similarity"] == 1.0]
    if exact:
        return {"created": False, "type": exact[0], "similar": [], "hint": "Exact match exists; reuse it."}
    if existing and not force:
        return {"created": False, "type": None, "similar": existing,
                "hint": "Similar types exist. Reuse one (or pass force=True if truly distinct)."}

    alias_list = sorted({_normalize(kind, a) for a in (aliases or [])} - {canonical})
    db.write(
        "CREATE (t:EntityType {key:$key, name:$name, kind:$kind, layer:'extension', description:$desc, "
        "example_props:$props, aliases:$aliases, builtin:false, created_at:$now, base:$base})",
        {"key": _key(kind, canonical), "name": canonical, "kind": kind, "desc": description,
         "props": example_props or [], "aliases": alias_list, "now": datetime.now(timezone.utc),
         "base": base if kind == "node" else None},
    )
    created = next(t for t in list_types(db, kind) if t["name"] == canonical)
    return {"created": True, "type": created, "similar": [], "hint": "New type registered."}


def is_registered(db: LumaDB, name: str, kind: str) -> bool:
    rows = db.read("MATCH (t:EntityType {key:$key}) RETURN count(t) AS n", {"key": _key(kind, name)})
    return rows[0]["n"] > 0


def get_type(db: LumaDB, name: str, kind: str) -> dict | None:
    """The registered type with exactly this (already normalised) name, or None."""
    return next((t for t in list_types(db, kind) if t["name"] == name), None)