import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from agents.tool_context import ToolContext

from app.agent.context import LumaContext
from app.agent.tools import graph_tools, memory_tools
from app.agent.tools.base import collect_tools

UTC = timezone.utc
TOOLS = {t.name: t for t in collect_tools(graph_tools, memory_tools)}


def run(db, tool, /, **args) -> dict:
    """Invoke a tool exactly as the Agents SDK does: a ToolContext plus a JSON argument string."""
    payload = json.dumps(args)
    ctx = ToolContext(context=LumaContext(db=db), tool_name=tool, tool_call_id="call_test", tool_arguments=payload)
    raw = asyncio.run(TOOLS[tool].on_invoke_tool(ctx, payload))
    assert isinstance(raw, str), "tools must return JSON text, not a Python object"
    return json.loads(raw)


def ok(db, tool, /, **args):
    out = run(db, tool, **args)
    assert "error" not in out, out
    return out


def err(db, tool, /, **args):
    out = run(db, tool, **args)
    assert "error" in out, f"expected an error, got {out}"
    return out["error"]


@pytest.fixture
def clock(monkeypatch):
    """Freeze 'now' so time-based tools are deterministic."""
    now = {"t": datetime(2026, 11, 21, 9, 0, tzinfo=UTC)}
    monkeypatch.setattr("app.agent.tools.base.utcnow", lambda: now["t"])
    return now


# ---------- plumbing ----------
def test_all_tools_are_registered():
    assert set(TOOLS) == {"list_types", "define_type", "create_node", "update_node", "create_relationship",
                          "find_entities", "get_node", "run_read_query", "add_person", "record_sighting",
                          "last_seen", "add_appointment", "set_reminder", "list_upcoming", "due_reminders",
                          "complete_reminder"}


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_tool_schema_is_documented_and_hides_internals(name):
    tool = TOOLS[name]
    props = tool.params_json_schema["properties"]
    assert tool.description and len(tool.description) > 20
    assert "ctx" not in props and "db" not in props          # injected by the SDK, never shown to the model
    for pname, spec in props.items():
        assert spec.get("description"), f"{name}.{pname} has no description (add it to the docstring Args)"
    for req in tool.params_json_schema.get("required", []):
        assert req in props


def test_bad_arguments_come_back_as_readable_errors(db):
    msg = err(db, "create_node", label="Object")                       # missing required `name`
    assert "Invalid arguments for create_node" in msg and "missing required: name" in msg
    msg = err(db, "find_entities", query="x", limit="many")            # wrong type
    assert "wrong type: limit (expected integer)" in msg
    msg = err(db, "define_type", name="X", kind="widget", description="d")   # not an allowed option
    assert "kind" in msg and "node" in msg and "relationship" in msg
    secret = "my-secret-medication-name"
    assert secret not in err(db, "find_entities", query=secret, limit="many")   # values are never echoed


def test_unexpected_failures_are_contained(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(graph_tools, "find_by_name", boom)
    msg = err(db, "create_node", label="Object", name="x")
    assert "Internal error" in msg and "disk on fire" not in msg        # no internals leaked to the model


# ---------- schema tools ----------
def test_list_types_hides_pipeline_layers(db):
    names = {t["name"] for t in ok(db, "list_types")["types"]}
    assert {"Person", "Object", "KNOWS"} <= names
    assert not names & {"Image", "Detection", "CONTAINS"}


def test_agent_can_extend_the_schema_then_use_it(db):
    err(db, "create_node", label="Vehicle", name="bike")                         # unregistered -> told what to do
    ok(db, "define_type", name="Vehicle", kind="node", description="A bike, car or bus", base="Entity")
    assert ok(db, "create_node", label="Vehicle", name="red bike")["created"]
    ok(db, "define_type", name="Meal", kind="node", description="A meal eaten", base="Event")
    meal = ok(db, "create_node", label="meal", name="lunch")["node"]           # label gets normalised
    assert db.read("MATCH (n:Event:Meal {id:$id}) RETURN count(n) AS c", {"id": meal["id"]})[0]["c"] == 1


# ---------- create / update ----------
def test_create_node_deduplicates_things(db):
    first = ok(db, "create_node", label="Object", name="Glasses")
    again = ok(db, "create_node", label="object", name="glasses")
    assert first["created"] and not again["created"] and again["node"]["id"] == first["node"]["id"]
    assert ok(db, "create_node", label="Object", name="glasses", allow_duplicate=True)["created"]


def test_locked_layers_and_bad_properties_are_refused(db):
    assert "locked" in err(db, "create_node", label="Image", name="x")
    assert "locked" in err(db, "create_node", label="Detection", name="x")
    assert "nested" in err(db, "create_node", label="Object", name="x", properties={"a": {"b": 1}})
    assert "managed by Luma" in err(db, "create_node", label="Object", name="x", properties={"id": "hack"})
    assert "snake_case" in err(db, "create_node", label="Object", name="x", properties={"Bad Key": 1})


def test_update_node(db):
    n = ok(db, "create_node", label="Object", name="keys", properties={"colour": "silver"})["node"]
    updated = ok(db, "update_node", id=n["id"], properties={"colour": "gold", "brand": "ACME"})["node"]
    assert updated["colour"] == "gold" and updated["brand"] == "ACME" and updated["name"] == "keys"
    assert "No node" in err(db, "update_node", id="nope", properties={"a": 1})
    assert "managed by Luma" in err(db, "update_node", id=n["id"], properties={"created_at": "x"})


# ---------- relationships ----------
def test_create_relationship_rules(db):
    sara = ok(db, "add_person", name="Sara")["person"]
    mug = ok(db, "create_node", label="Object", name="mug")["node"]
    ok(db, "create_relationship", from_id=mug["id"], rel_type="owned by", to_id=sara["id"])
    ok(db, "create_relationship", from_id=mug["id"], rel_type="OWNED_BY", to_id=sara["id"])   # idempotent
    n = db.read("MATCH (:Object {id:$a})-[r:OWNED_BY]->() RETURN count(r) AS c", {"a": mug["id"]})[0]["c"]
    assert n == 1
    assert "not registered" in err(db, "create_relationship", from_id=mug["id"], rel_type="HATES", to_id=sara["id"])
    assert "perception pipeline" in err(db, "create_relationship", from_id=mug["id"], rel_type="CONTAINS", to_id=sara["id"])
    assert "No node" in err(db, "create_relationship", from_id="ghost", rel_type="KNOWS", to_id=sara["id"])
    ok(db, "define_type", name="allergic to", kind="relationship", description="Allergy")
    ok(db, "create_relationship", from_id=sara["id"], rel_type="ALLERGIC_TO", to_id=mug["id"], properties={"severity": "mild"})


# ---------- search ----------
def test_find_entities_exact_alias_and_text(db):
    ok(db, "add_person", name="Dr. Karim Ben Salah", relationship="doctor", aliases=["Dr Karim"])
    ok(db, "create_node", label="Object", name="blue umbrella")
    assert ok(db, "find_entities", query="dr karim")["matches"][0]["match"] == "exact"
    umbrella = ok(db, "find_entities", query="umbrella")["matches"]
    assert umbrella and umbrella[0]["match"] == "text"
    assert ok(db, "find_entities", query="umbrella", label="Person")["matches"] == []
    assert ok(db, "find_entities", query="o'clock (weird) AND:")["matches"] == []       # no crash on odd input


def test_get_node_shows_connections(db):
    sara = ok(db, "add_person", name="Sara", relationship="sister")["person"]
    conns = ok(db, "get_node", id=sara["id"])["connections"]
    assert {"direction": "in", "rel": "KNOWS", "node": {"id": "self", "name": "Me", "_labels": ["Entity", "Person"]}} in conns


def test_run_read_query_is_read_only_and_capped(db):
    for i in range(5):
        ok(db, "create_node", label="Object", name=f"thing {i}")
    out = ok(db, "run_read_query", cypher="MATCH (o:Object) RETURN o.name AS n ORDER BY n", max_rows=3)
    assert len(out["rows"]) == 3 and out["truncated"]
    assert "error" in run(db, "run_read_query", cypher="MATCH (o:Object) DETACH DELETE o")
    assert "error" in run(db, "run_read_query", cypher="MATCH (o RETURN")                  # syntax error is reported
    assert ok(db, "run_read_query", cypher="MATCH (o:Object) RETURN count(o) AS c")["rows"] == [{"c": 5}]


def test_embeddings_never_reach_the_agent(db):
    n = ok(db, "create_node", label="Object", name="cup", properties={"embedding": [0.1, 0.2, 0.3, 0.4]})["node"]
    assert "embedding" not in n
    out = ok(db, "run_read_query", cypher="MATCH (o:Object {name:'cup'}) RETURN o")
    assert "embedding" not in out["rows"][0]["o"]


# ---------- people ----------
def test_add_person_merges_instead_of_duplicating(db):
    a = ok(db, "add_person", name="Sara", relationship="sister", phone="111")
    b = ok(db, "add_person", name="sara", email="s@x.com", aliases=["Sarah"])
    assert a["created"] and not b["created"] and a["person"]["id"] == b["person"]["id"]
    assert b["person"]["phone"] == "111" and b["person"]["email"] == "s@x.com"
    rel = db.read("MATCH (:Person {id:'self'})-[r:KNOWS]->(:Person {name:'Sara'}) RETURN r.relationship AS r")
    assert rel == [{"r": "sister"}]
    assert ok(db, "find_entities", query="Sarah")["matches"][0]["match"] == "exact"


# ---------- sightings ----------
def test_last_seen_returns_newest_first(db, clock):
    ok(db, "record_sighting", thing="glasses", place="sofa", at="2026-11-20T19:00:00+00:00")
    ok(db, "record_sighting", thing="Glasses", place="kitchen", context="on the table next to a mug",
       at="2026-11-21T08:14:00+00:00")
    out = ok(db, "last_seen", thing="glasses")
    assert out["sightings"][0]["place"] == "kitchen" and out["sightings"][0]["context"] == "on the table next to a mug"
    assert [s["place"] for s in out["sightings"]] == ["kitchen", "sofa"]
    assert db.read("MATCH (o:Object) RETURN count(o) AS c")[0]["c"] == 1          # one Object, many sightings


def test_last_seen_unknown_and_never_seen(db):
    ok(db, "create_node", label="Object", name="wallet")
    assert "never been seen" in ok(db, "last_seen", thing="wallet")["hint"]
    ok(db, "create_node", label="Object", name="red umbrella")
    out = ok(db, "last_seen", thing="umbrella")
    assert out["found"] is False and "red umbrella" in out["did_you_mean"]


def test_sighting_with_people_and_bad_label(db):
    ok(db, "add_person", name="Sara")
    out = ok(db, "record_sighting", thing="keys", place="hall", with_people=["Sara", "Stranger"])
    assert out["people_not_known"] == ["Stranger"]
    who = db.read("MATCH (:Sighting {id:$id})-[:WITH]->(p:Person) RETURN p.name AS n", {"id": out["sighting"]["id"]})
    assert who == [{"n": "Sara"}]
    assert "must be a registered persistent type" in err(db, "record_sighting", thing="x", thing_label="Appointment")


# ---------- appointments & reminders ----------
def test_appointment_with_reminder_flow(db, clock):
    out = ok(db, "add_appointment", title="Dentist", start="2026-11-24T15:30:00+00:00", place="Clinic",
             with_person="Dr Karim", remind_minutes_before=60)
    assert out["reminder"]["due"] == "2026-11-24T14:30:00+00:00"   # tool results are JSON: times are ISO strings
    upcoming = ok(db, "list_upcoming", days=7)
    assert upcoming["appointments"][0]["title"] == "Dentist"
    assert upcoming["appointments"][0]["place"] == "Clinic" and upcoming["appointments"][0]["people"] == ["Dr Karim"]
    assert len(upcoming["reminders"]) == 1
    assert ok(db, "list_upcoming", days=1)["appointments"] == []                  # outside the window
    assert db.read("MATCH (:Reminder)-[:ABOUT]->(:Appointment) RETURN count(*) AS c")[0]["c"] == 1


def test_reminders_become_due_and_can_be_completed(db, clock):
    r = ok(db, "set_reminder", text="Take ibuprofen", due="2026-11-21T12:00:00+00:00")["reminder"]
    assert ok(db, "due_reminders")["reminders"] == []
    clock["t"] = datetime(2026, 11, 21, 12, 5, tzinfo=UTC)
    assert [x["id"] for x in ok(db, "due_reminders")["reminders"]] == [r["id"]]
    ok(db, "complete_reminder", id=r["id"])
    assert ok(db, "due_reminders")["reminders"] == []
    assert "No reminder" in err(db, "complete_reminder", id="nope")


def test_time_validation(db, clock):
    assert "Could not read time" in err(db, "set_reminder", text="x", due="tomorrow at 3")
    assert "end is before start" in err(db, "add_appointment", title="x", start="2026-11-24T15:00", end="2026-11-24T14:00")
    out = ok(db, "set_reminder", text="x", due="2026-11-22T10:00")                 # no offset -> configured zone (UTC default)
    assert out["reminder"]["due"] == "2026-11-22T10:00:00+00:00"
    about = ok(db, "create_node", label="Object", name="pills")["node"]
    ok(db, "set_reminder", text="refill", due="2026-11-22T10:00", about_id=about["id"])
    assert "No node" in err(db, "set_reminder", text="x", due="2026-11-22T10:00", about_id="ghost")