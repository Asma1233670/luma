import http.client
import json
import re
import threading
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

import pytest

from app.agent.tools.graph_tools import ensure_entity, insert_node, link_known
from app.agent.tools.memory_tools import upsert_person
from app.viz.__main__ import main as viz_main
from app.viz.__main__ import make_handler
from app.viz.graph_export import classify, clean_value, display_name, read_graph
from app.viz.html_export import render_html


@pytest.fixture
def populated(db):
    """A small but realistic memory: a sister, glasses seen in the kitchen, an appointment."""
    sara = upsert_person(db, "Sara", relationship="sister", phone="555-0100")["person"]
    glasses = ensure_entity(db, "Object", "glasses")
    kitchen = ensure_entity(db, "Place", "kitchen")
    s = insert_node(db, "Sighting", "Event", "glasses seen",
                    {"at": datetime(2026, 11, 21, 8, 14, tzinfo=timezone.utc), "context": "on the table"})
    link_known(db, s["id"], "Event", "OF", glasses["id"], "Entity")
    link_known(db, s["id"], "Event", "IN", kitchen["id"], "Entity")
    link_known(db, s["id"], "Event", "WITH", sara["id"], "Entity")
    return db


# ---------- classification & cleaning ----------
def test_classify():
    assert classify(["Entity", "Person"]) == ("Person", "entity")
    assert classify(["Event", "Sighting"]) == ("Sighting", "event")
    assert classify(["EntityType"]) == ("EntityType", "registry")
    assert classify(["Image"]) == ("Image", "media")
    assert classify(["Detection"]) == ("Detection", "perception")
    assert classify(["Entity", "Vehicle"]) == ("Vehicle", "entity")     # agent-defined types work too
    assert classify([]) == ("Unlabeled", "other")


def test_display_names():
    assert display_name("Person", {"name": "Sara"}) == "Sara"
    assert display_name("Reminder", {"text": "Take ibuprofen twice daily with plenty of water after eating"}).endswith("…")
    assert display_name("Thing", {}) == "Thing"


def test_values_are_json_safe_and_embeddings_are_hidden():
    assert clean_value("embedding", [0.1] * 512) == "[512 numbers, hidden]"
    assert clean_value("at", datetime(2026, 11, 21, 8, 14, tzinfo=timezone.utc)) == "2026-11-21T08:14:00+00:00"
    assert len(clean_value("text", "x" * 5000)) <= 600
    json.dumps(clean_value("odd", object()))                              # anything unusual becomes text


# ---------- reading the graph ----------
def test_read_graph_returns_the_memory(populated):
    g = read_graph(populated.path)
    by_name = {n["name"]: n for n in g["nodes"]}
    assert by_name["Sara"]["layer"] == "entity" and by_name["Sara"]["props"]["phone"] == "555-0100"
    assert by_name["glasses seen"]["layer"] == "event"
    assert by_name["glasses seen"]["props"]["at"] == "2026-11-21T08:14:00+00:00"
    types = sorted(e["type"] for e in g["edges"])
    assert types == ["IN", "KNOWS", "OF", "WITH"]
    ids = {n["id"] for n in g["nodes"]}
    assert all(e["from"] in ids and e["to"] in ids for e in g["edges"])
    assert g["stats"]["truncated"] is False
    json.dumps(g)                                                          # the whole snapshot is JSON-serialisable


def test_reading_works_while_the_app_holds_the_writer(populated):
    # `populated` still has its writer open and never flushed: the viewer must still see everything
    assert any(n["name"] == "Sara" for n in read_graph(populated.path)["nodes"])


def test_viewer_cannot_modify_the_database(populated):
    before = read_graph(populated.path)["stats"]["nodes"]
    read_graph(populated.path)
    assert read_graph(populated.path)["stats"]["nodes"] == before


def test_big_graphs_keep_the_meaningful_layers(populated):
    g = read_graph(populated.path, max_nodes=5)      # the sample has exactly 5 world nodes (4 things + 1 event)
    assert g["stats"]["truncated"] and g["stats"]["nodes"] == 5 and g["stats"]["total_nodes"] > 5
    assert {n["layer"] for n in g["nodes"]} == {"entity", "event"}         # registry/media/perception are dropped first


# ---------- the HTML page ----------
def test_html_is_self_contained_and_has_the_data(populated):
    html = render_html(read_graph(populated.path))
    assert "/*__VIS_JS__*/" not in html and "/*__GRAPH_DATA__*/" not in html
    assert "Sara" in html and "vis-network" in html
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)         # nothing is fetched from the internet
    assert "cdn" not in html.lower().split("<script>")[2]                  # (app script) no CDN references


def test_memory_text_cannot_break_out_of_the_page(db):
    evil = '</script><script>window.pwned=1</script><!-- \u2028'
    insert_node(db, "Memory", "Event", evil, {"text": evil})
    html = render_html(read_graph(db.path))
    assert html.count("</script>") == 2                                    # only the template's own two closings
    assert "<script>window.pwned" not in html and "\\u003c/script>" in html


# ---------- live server ----------
@pytest.fixture
def server(populated):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(populated.path, 3000))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def get(srv, path, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    conn.request("GET", path, headers={"Host": host or f"127.0.0.1:{srv.server_port}"})
    r = conn.getresponse()
    return r.status, r.read(), r.getheader("Content-Type")


def test_server_serves_page_and_json(server, populated):
    status, body, ctype = get(server, "/")
    assert status == 200 and "text/html" in ctype and b"Luma memory graph" in body
    status, body, ctype = get(server, "/graph.json")
    assert status == 200 and "json" in ctype and any(n["name"] == "Sara" for n in json.loads(body)["nodes"])
    upsert_person(populated, "Karim", relationship="doctor")               # the page is live: new memories appear
    assert any(n["name"] == "Karim" for n in json.loads(get(server, "/graph.json")[1])["nodes"])
    assert get(server, "/nope")[0] == 404


def test_server_refuses_foreign_hosts(server):
    assert get(server, "/graph.json", host="evil.example.com")[0] == 403


# ---------- command line ----------
def test_cli_writes_a_snapshot(populated, tmp_path, capsys):
    out = tmp_path / "g.html"
    assert viz_main(["--db", populated.path, "--out", str(out)]) == 0
    assert out.exists() and "Sara" in out.read_text(encoding="utf-8")
    assert "don't commit or share" in capsys.readouterr().out


def test_cli_explains_a_missing_database(tmp_path, capsys):
    assert viz_main(["--db", str(tmp_path / "nope.relate")]) == 1
    assert "No database at" in capsys.readouterr().err