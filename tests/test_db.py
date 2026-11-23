import threading

import pytest
import relate

from app.db.bootstrap import bootstrap
from app.db.client import LumaDB
from app.db.guards import InvalidName, normalize_label, normalize_rel_type, validate_label, validate_rel_type
from app.db.registry import BUILTIN_TYPES, define_type, is_registered, list_types

DIM = 4


# ---------- bootstrap ----------
def test_bootstrap_is_idempotent(db):
    bootstrap(db, embed_dim=DIM)  # second run must not fail or duplicate
    assert len(list_types(db)) == len(BUILTIN_TYPES)


def test_ids_are_unique(db):
    db.write("CREATE (:Entity:Object {id:'o1', name:'glasses'})")
    with pytest.raises(relate.IntegrityError):
        db.write("CREATE (:Entity:Person {id:'o1', name:'Sara'})")


# ---------- read-only guarantee ----------
def test_reads_see_writes_and_cannot_write(db):
    db.write("CREATE (:Entity:Object {id:'o1', name:'glasses'})")
    assert db.read("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"] == 1
    with pytest.raises(relate.OperationalError):
        db.read("CREATE (:Entity {id:'x'})")
    with pytest.raises(relate.OperationalError):
        db.read("MATCH (n) DETACH DELETE n")
    assert db.read("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"] == 1


def test_nodes_come_back_as_plain_dicts(db):
    db.write("CREATE (:Entity:Person {id:'p1', name:'Sara'})")
    node = db.read("MATCH (p:Person) RETURN p")[0]["p"]
    assert node["name"] == "Sara" and set(node["_labels"]) == {"Entity", "Person"}


def test_read_capped_flags_truncation(db):
    for i in range(5):
        db.write("CREATE (:Entity {id:$i})", {"i": f"e{i}"})
    out = db.read_capped("MATCH (e:Entity) RETURN e.id AS id", max_rows=3)
    assert len(out["rows"]) == 3 and out["truncated"] and out["total"] == 5


def test_transaction_rolls_back(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.write("CREATE (:Entity {id:'tmp'})")
            raise RuntimeError("boom")
    assert db.read("MATCH (e:Entity) RETURN count(e) AS n")[0]["n"] == 0


# ---------- registry ----------
def test_normalisation():
    assert normalize_label("eye glasses") == "EyeGlasses"
    assert normalize_rel_type("last seen in") == "LAST_SEEN_IN"
    assert normalize_rel_type("lastSeenIn") == "LAST_SEEN_IN"


def test_names_cannot_smuggle_cypher():
    # Raw names with Cypher syntax are rejected outright...
    for bad in ("Person) DETACH DELETE n //", "a b", "lower", "X"):
        with pytest.raises(InvalidName):
            validate_label(bad)
    with pytest.raises(InvalidName):
        validate_rel_type("KNOWS]->() DETACH DELETE n //")
    # ...and normalisation strips everything except letters/digits.
    cleaned = normalize_label("x); MATCH (n) DETACH DELETE n //")
    assert cleaned.isalnum()


def test_define_new_type(db):
    r = define_type(db, "vehicle", "node", "A car, bike or bus", ["plate"], aliases=["car"])
    assert r["created"] and r["type"]["name"] == "Vehicle" and not r["type"]["builtin"]
    assert is_registered(db, "Vehicle", "node")


def test_define_type_reuses_exact_match(db):
    r = define_type(db, "person", "node", "dup")
    assert not r["created"] and r["type"]["name"] == "Person"


def test_define_type_treats_spacing_and_case_as_same_type(db):
    define_type(db, "Eyeglasses", "node", "Glasses worn on the face")
    r = define_type(db, "eye glasses", "node", "same thing")  # normalises to EyeGlasses
    assert not r["created"] and r["type"]["name"] == "Eyeglasses"


def test_define_type_flags_near_duplicates(db):
    define_type(db, "Eyeglasses", "node", "Glasses worn on the face")
    r = define_type(db, "Eyeglass", "node", "singular form")
    assert not r["created"] and r["type"] is None and r["similar"][0]["name"] == "Eyeglasses"
    assert define_type(db, "Eyeglass", "node", "singular form", force=True)["created"]


def test_locked_labels_cannot_be_defined(db):
    for locked in ("Image", "Detection", "Entity", "EntityType"):
        with pytest.raises(InvalidName):
            define_type(db, locked, "node", "x")


def test_relationship_types(db):
    r = define_type(db, "allergic to", "relationship", "Person is allergic to something")
    assert r["created"] and r["type"]["name"] == "ALLERGIC_TO"
    assert define_type(db, "KNOWS", "relationship", "dup")["created"] is False


# ---------- the schema doing real work ----------
def test_last_seen_query_on_durable_layer(db):
    db.write("""
        CREATE (o:Entity:Object {id:'o1', name:'glasses'}),
               (k:Entity:Place  {id:'pl1', name:'kitchen'}),
               (s1:Event:Sighting {id:'s1', at: datetime('2026-11-21T08:14:00Z'), context:'on the kitchen table'}),
               (s2:Event:Sighting {id:'s2', at: datetime('2026-11-20T19:00:00Z'), context:'on the sofa'}),
               (s1)-[:OF]->(o), (s2)-[:OF]->(o), (s1)-[:IN]->(k)
    """)
    row = db.read(
        "MATCH (s:Sighting)-[:OF]->(:Object {name:$name}) OPTIONAL MATCH (s)-[:IN]->(p:Place) "
        "RETURN p.name AS place, s.context AS context ORDER BY s.at DESC LIMIT 1",
        {"name": "glasses"},
    )[0]
    assert row == {"place": "kitchen", "context": "on the kitchen table"}


def test_vector_search_with_traversal(db):
    db.write("""
        CREATE (o:Entity:Object {id:'o1', name:'glasses', embedding:[1.0,0.0,0.0,0.0]}),
               (m:Entity:Object {id:'o2', name:'mug',     embedding:[0.0,1.0,0.0,0.0]}),
               (k:Entity:Place  {id:'pl1', name:'kitchen'}),
               (o)-[:LOCATED_AT]->(k)
    """)
    rows = db.read(
        "CALL db.index.vector.queryNodes('entity_embedding', 1, $q) YIELD node, score "
        "MATCH (node)-[:LOCATED_AT]->(p:Place) RETURN node.name AS name, p.name AS place",
        {"q": [0.9, 0.1, 0.0, 0.0]},
    )
    assert rows == [{"name": "glasses", "place": "kitchen"}]


def test_fulltext_over_ocr_text(db):
    db.write("CREATE (:TextBlock {id:'t1', text:'Take one tablet of ibuprofen twice daily'})")
    rows = db.read("CALL db.index.fulltext.queryNodes('textblock_text','ibuprofen') YIELD node RETURN node.id AS id")
    assert rows == [{"id": "t1"}]


# ---------- threading (Relate connections are thread-affine) ----------

def _in_thread(fn):
    box = {}

    def run():
        try:
            box["ok"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=run)
    t.start()
    t.join()
    if "err" in box:
        raise box["err"]
    return box["ok"]


def test_db_works_from_any_thread(db):
    _in_thread(lambda: db.write("CREATE (:Entity {id:'from_thread'})"))
    assert _in_thread(lambda: db.read("MATCH (e:Entity) RETURN count(e) AS c"))[0]["c"] == 1

    def txn():
        with db.transaction():
            db.write("CREATE (:Entity {id:'t1'})")
    _in_thread(txn)
    assert db.read("MATCH (e:Entity) RETURN count(e) AS c")[0]["c"] == 2


def test_concurrent_writers_do_not_conflict(db):
    errors = []

    def work(i):
        try:
            db.write("CREATE (:Entity {id:$i})", {"i": f"w{i}"})
            db.read("MATCH (e:Entity) RETURN count(e) AS c")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert db.read("MATCH (e:Entity) RETURN count(e) AS c")[0]["c"] == 20


def test_transactions_from_different_threads_do_not_interleave(db):
    started, release = threading.Event(), threading.Event()

    def slow_txn():
        with db.transaction():
            db.write("CREATE (:Entity {id:'a1'})")
            started.set()
            release.wait(2)
            db.write("CREATE (:Entity {id:'a2'})")

    t = threading.Thread(target=slow_txn)
    t.start()
    started.wait(2)
    other = threading.Thread(target=lambda: db.write("CREATE (:Entity {id:'b1'})"))
    other.start()
    other.join(0.3)
    assert other.is_alive()          # blocked until the transaction finishes
    release.set()
    t.join()
    other.join()
    assert db.read("MATCH (e:Entity) RETURN count(e) AS c")[0]["c"] == 3


def test_close_is_idempotent(tmp_path):
    d = LumaDB(str(tmp_path / "x.relate"))
    d.close()
    d.close()