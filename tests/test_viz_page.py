"""Runs the real exported page in a simulated browser (jsdom). Optional: skipped unless Node and jsdom exist.

One-time setup to enable it:   cd tests/viz_page && npm install
If installing on the Windows drive from WSL gives a damaged jsdom, install it in your Linux home instead and
point to it:   mkdir -p ~/luma-viz && cp tests/viz_page/package.json ~/luma-viz && (cd ~/luma-viz && npm install)
               export LUMA_JSDOM_DIR=~/luma-viz
"""
import copy
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.agent.tools.graph_tools import ensure_entity, insert_node, link_known
from app.agent.tools.memory_tools import upsert_person
from app.viz.graph_export import read_graph
from app.viz.html_export import render_html

CHECK_DIR = Path(__file__).parent / "viz_page"
JSDOM_ROOT = Path(os.environ.get("LUMA_JSDOM_DIR", CHECK_DIR)).expanduser()

pytestmark = pytest.mark.skipif(
    not shutil.which("node") or not (JSDOM_ROOT / "node_modules" / "jsdom").exists(),
    reason="needs Node.js and jsdom: run `npm install` in tests/viz_page (optional browser-level check)",
)


@pytest.fixture
def page(db, tmp_path):
    sara = upsert_person(db, "Sara", relationship="sister", phone="555-0100")["person"]
    glasses = ensure_entity(db, "Object", "glasses")
    kitchen = ensure_entity(db, "Place", "kitchen")
    s = insert_node(db, "Sighting", "Event", "glasses seen",
                    {"at": datetime(2026, 11, 21, 8, 14, tzinfo=timezone.utc), "context": "on the table"})
    for rel, target in (("OF", glasses), ("IN", kitchen), ("WITH", sara)):
        link_known(db, s["id"], "Event", rel, target["id"], "Entity")
    insert_node(db, "Memory", "Event", '<img src=x onerror="window.__pwned=1">',
                {"text": "<script>window.__pwned=1</script>"})          # hostile text must stay inert
    graph = read_graph(db.path)

    after = copy.deepcopy(graph)                                          # a later snapshot: Karim learned, glasses gone
    gone = next(n for n in after["nodes"] if n["name"] == "glasses")
    after["nodes"] = [n for n in after["nodes"] if n["id"] != gone["id"]]
    after["edges"] = [e for e in after["edges"] if gone["id"] not in (e["from"], e["to"])]
    me = next(n["id"] for n in after["nodes"] if n["name"] == "Me")
    after["nodes"].append({"id": 99999, "labels": ["Entity", "Person"], "kind": "Person", "layer": "entity",
                           "name": "Karim", "props": {"name": "Karim"}})
    after["edges"].append({"id": 99998, "from": 99999, "to": me, "type": "KNOWS", "props": {}})

    files = {"html": tmp_path / "graph.html", "graph": tmp_path / "graph.json", "after": tmp_path / "after.json"}
    files["html"].write_text(render_html(graph), encoding="utf-8")
    files["graph"].write_text(json.dumps(graph))
    files["after"].write_text(json.dumps(after))
    return files


def run_check(*args):
    r = subprocess.run(["node", "check.mjs", *map(str, args)], cwd=CHECK_DIR, capture_output=True, text=True,
                       timeout=90, env={**os.environ, "LUMA_JSDOM_DIR": str(JSDOM_ROOT)})
    if r.returncode == 3:  # the tool is broken, not Luma: say so instead of failing the whole suite
        pytest.skip(r.stderr.strip() + "  -> delete node_modules and reinstall, or see the note at the top of this file")
    assert r.returncode == 0, r.stdout + r.stderr


def test_snapshot_page_works_in_a_browser(page):
    run_check(page["html"], page["graph"])


def test_served_page_updates_live(page):
    run_check(page["html"], page["graph"], "live", page["after"])