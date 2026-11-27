"""Luma's everyday memory: people, where things were seen, appointments, reminders."""
from datetime import timedelta

from agents import RunContextWrapper

from backend.app.agent.context import LumaContext
from backend.app.agent.tools import base
from backend.app.agent.tools.base import ToolError, clean_props, luma_tool, parse_time
from backend.app.agent.tools.graph_tools import (ensure_entity, find_by_name, insert_node, link_known, public,
                                         require_node, similar_entities)
from backend.app.db import registry
from backend.app.db.client import LumaDB
from backend.app.db.guards import normalize_label

SELF_ID = "self"


def ensure_self(db: LumaDB) -> None:
    """The user is a Person node with a fixed id, so 'my sister' uses the same pattern as everything else."""
    now = base.utcnow()
    db.write("MERGE (n:Entity:Person {id:$id}) ON CREATE SET n.name='Me', n.created_at=$now, n.updated_at=$now",
             {"id": SELF_ID, "now": now})


# ---- people ---------------------------------------------------------------------------------
def upsert_person(db: LumaDB, name: str, relationship: str | None = None, phone: str | None = None,
                  email: str | None = None, notes: str | None = None, aliases: list[str] | None = None) -> dict:
    ensure_self(db)
    details = clean_props({"phone": phone, "email": email, "notes": notes}, frozenset())
    existing = find_by_name(db, "Person", name)
    if existing:
        merged = sorted(set(existing.get("aliases") or []) | set(aliases or []))
        props = {**details, **({"aliases": merged} if merged else {})}
        person = existing
        if props:
            person = db.write("MATCH (n:Entity:Person {id:$id}) SET n += $p, n.updated_at=$now RETURN n",
                              {"id": existing["id"], "p": props, "now": base.utcnow()})[0]["n"]
        created = False
    else:
        person = insert_node(db, "Person", "Entity", name.strip(),
                             {**details, **({"aliases": aliases} if aliases else {})})
        created = True
    if relationship:
        link_known(db, SELF_ID, "Entity", "KNOWS", person["id"], "Entity", {"relationship": relationship.strip()})
    return {"created": created, "person": public(person)}


@luma_tool
def add_person(ctx: RunContextWrapper[LumaContext], name: str, relationship: str | None = None,
               phone: str | None = None, email: str | None = None, notes: str | None = None,
               aliases: list[str] | None = None) -> dict:
    """Remember a person the user knows. If the person is already known, the given details are
    merged into the existing record instead of creating a duplicate.

    Args:
        name: The person's name.
        relationship: Who they are TO THE USER, e.g. 'sister', 'doctor', 'neighbour'.
        phone: Phone number.
        email: Email address.
        notes: Anything else worth remembering.
        aliases: Other names or nicknames, e.g. ['Sarah'].
    """
    return upsert_person(ctx.context.db, name, relationship, phone, email, notes, aliases)


# ---- where things were seen -----------------------------------------------------------------
@luma_tool
def record_sighting(ctx: RunContextWrapper[LumaContext], thing: str, place: str | None = None,
                    context: str | None = None, at: str | None = None, thing_label: str = "Object",
                    with_people: list[str] | None = None, confidence: float = 0.9) -> dict:
    """Note that something was seen or left somewhere, from what the user says or what you observe.
    Creates the thing and the place if they are new. Use for 'I left my glasses in the kitchen'.

    Args:
        thing: What was seen, e.g. 'glasses'.
        place: Where, e.g. 'kitchen'.
        context: Short detail, e.g. 'on the table next to a mug'.
        at: ISO 8601 time of the sighting. Omit for now.
        thing_label: Type of the thing. Default 'Object'; use 'Person' for a person.
        with_people: Names of known people who were present.
        confidence: 0 to 1. Lower it when the user is unsure.
    """
    db = ctx.context.db
    label = normalize_label(thing_label)
    t = registry.get_type(db, label, "node")
    if t is None or t["base"] != "Entity":
        raise ToolError(f"thing_label {label!r} must be a registered persistent type (see list_types).")
    when = parse_time(at) if at else base.utcnow()
    entity = ensure_entity(db, label, thing)
    where = ensure_entity(db, "Place", place) if place else None
    people, unknown = [], []
    for n in with_people or []:
        p = find_by_name(db, "Person", n)
        if p:
            people.append(p)
        else:
            unknown.append(n)
    props = {"at": when, "confidence": float(confidence), **({"context": context} if context else {})}
    with db.transaction():
        s = insert_node(db, "Sighting", "Event", f"{entity['name']} seen", props)
        link_known(db, s["id"], "Event", "OF", entity["id"], "Entity")
        if where:
            link_known(db, s["id"], "Event", "IN", where["id"], "Entity")
        for p in people:
            link_known(db, s["id"], "Event", "WITH", p["id"], "Entity")
    return {"sighting": s, "thing": entity, "place": where, "people_not_known": unknown}


@luma_tool
def last_seen(ctx: RunContextWrapper[LumaContext], thing: str, limit: int = 3) -> dict:
    """Where was something last seen? Returns the most recent sightings, newest first, with place and
    context. `at` is the last moment it was seen there and `first_seen_here` when it first appeared there.
    Matches the exact name or an alias; if nothing matches you get close names back.

    Args:
        thing: What to look for, e.g. 'glasses'.
        limit: How many sightings to return (1-10).
    """
    db = ctx.context.db
    entity = find_by_name(db, None, thing)
    if entity is None:
        candidates = similar_entities(db, thing)
        names = [c["name"] for c in candidates]
        hint = f"Nothing remembered called {thing!r}."
        if names:
            hint += " Did you mean " + ", ".join(repr(n) for n in names) + "?"
        return {"found": False, "did_you_mean": names, "candidates": candidates, "hint": hint}
    rows = db.read(
        "MATCH (s:Sighting)-[:OF]->(e:Entity {id:$id}) OPTIONAL MATCH (s)-[:IN]->(p:Place) "
        "RETURN coalesce(s.last_seen, s.at) AS at, s.at AS first_seen_here, p.name AS place, "
        "s.context AS context, s.confidence AS confidence ORDER BY at DESC LIMIT $limit",
        {"id": entity["id"], "limit": max(1, min(limit, 10))})
    return {"found": True, "thing": entity["name"], "sightings": rows,
            **({} if rows else {"hint": f"{entity['name']} is known but has never been seen anywhere."})}


# ---- appointments & reminders ---------------------------------------------------------------
@luma_tool
def add_appointment(ctx: RunContextWrapper[LumaContext], title: str, start: str, end: str | None = None,
                    place: str | None = None, with_person: str | None = None, notes: str | None = None,
                    remind_minutes_before: int | None = None) -> dict:
    """Store an appointment, optionally with a place, a person, and an automatic reminder.

    Args:
        title: What it is, e.g. 'Dentist'.
        start: ISO 8601 start time, e.g. 2026-11-24T15:30. Times without an offset use the user's zone.
        end: ISO 8601 end time.
        place: Where it takes place; created if new.
        with_person: Who it is with; created if unknown.
        notes: Extra details.
        remind_minutes_before: Create a reminder this many minutes before the start.
    """
    db = ctx.context.db
    begins = parse_time(start)
    ends = parse_time(end) if end else None
    if ends and ends < begins:
        raise ToolError("end is before start.")
    where = ensure_entity(db, "Place", place) if place else None
    person = None
    if with_person:
        person = find_by_name(db, "Person", with_person) or upsert_person(db, with_person)["person"]
    props = {"title": title, "start": begins, "at": begins, "status": "scheduled",
             **({"end": ends} if ends else {}), **({"notes": notes} if notes else {})}
    reminder = None
    with db.transaction():
        appt = insert_node(db, "Appointment", "Event", title, props)
        if where:
            link_known(db, appt["id"], "Event", "AT", where["id"], "Entity")
        if person:
            link_known(db, appt["id"], "Event", "WITH", person["id"], "Entity")
        if remind_minutes_before is not None:
            due = begins - timedelta(minutes=remind_minutes_before)
            reminder = insert_node(db, "Reminder", "Event", f"{title} soon",
                                   {"text": f"{title} in {remind_minutes_before} minutes", "due": due,
                                    "at": due, "status": "pending"})
            link_known(db, reminder["id"], "Event", "ABOUT", appt["id"], "Event")
    return {"appointment": appt, "reminder": reminder}


@luma_tool
def set_reminder(ctx: RunContextWrapper[LumaContext], text: str, due: str, about_id: str | None = None) -> dict:
    """Remind the user of something at a given time.

    Args:
        text: What to remind them of, e.g. 'Take ibuprofen'.
        due: ISO 8601 time. Times without an offset use the user's zone.
        about_id: Optional id of an existing node this reminder concerns.
    """
    db = ctx.context.db
    when = parse_time(due)
    about_base = require_node(db, about_id) if about_id else None
    with db.transaction():
        r = insert_node(db, "Reminder", "Event", text, {"text": text, "due": when, "at": when, "status": "pending"})
        if about_id:
            link_known(db, r["id"], "Event", "ABOUT", about_id, about_base)
    return {"reminder": r}


@luma_tool
def list_upcoming(ctx: RunContextWrapper[LumaContext], days: int = 7) -> dict:
    """What's coming up: scheduled appointments and pending reminders in the next N days.

    Args:
        days: How many days ahead to look (1-90).
    """
    db = ctx.context.db
    now = base.utcnow()
    window = {"now": now, "end": now + timedelta(days=max(1, min(days, 90)))}
    appts = db.read(
        "MATCH (a:Appointment) WHERE a.status='scheduled' AND a.start >= $now AND a.start <= $end "
        "OPTIONAL MATCH (a)-[:AT]->(p:Place) OPTIONAL MATCH (a)-[:WITH]->(w:Person) "
        "RETURN a.id AS id, a.title AS title, a.start AS start, a.end AS end, p.name AS place, "
        "collect(w.name) AS people ORDER BY start", window)
    reminders = db.read(
        "MATCH (r:Reminder) WHERE r.status='pending' AND r.due >= $now AND r.due <= $end "
        "RETURN r.id AS id, r.text AS text, r.due AS due ORDER BY due", window)
    return {"appointments": appts, "reminders": reminders}


def pending_reminders_due(db: LumaDB) -> list[dict]:
    """Reminders whose time has come. Also used by the scheduler, outside the agent."""
    return db.read("MATCH (r:Reminder) WHERE r.status='pending' AND r.due <= $now "
                   "RETURN r.id AS id, r.text AS text, r.due AS due ORDER BY due", {"now": base.utcnow()})


@luma_tool
def due_reminders(ctx: RunContextWrapper[LumaContext]) -> dict:
    """Pending reminders whose time has come and that the user has not been told about yet."""
    return {"reminders": pending_reminders_due(ctx.context.db)}


@luma_tool
def complete_reminder(ctx: RunContextWrapper[LumaContext], id: str) -> dict:
    """Mark a reminder as done once the user has been told about it.

    Args:
        id: The reminder's id (from due_reminders or list_upcoming).
    """
    now = base.utcnow()
    rows = ctx.context.db.write(
        "MATCH (r:Event:Reminder {id:$id}) SET r.status='done', r.done_at=$now, r.updated_at=$now "
        "RETURN r.id AS id", {"id": id, "now": now})
    if not rows:
        raise ToolError(f"No reminder with id {id!r}.")
    return {"completed": id}