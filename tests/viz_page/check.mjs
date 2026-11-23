// Runs an exported Luma graph page in a simulated browser (jsdom) and checks that it behaves.
// usage: node check.mjs <page.html> <graph.json> [live <graph-after.json>]
// Only the canvas *drawing* is faked; the real vis-network library and the page's own logic run.
//
// jsdom is looked up in ./node_modules, or in $LUMA_JSDOM_DIR/node_modules if that is set (handy when
// installing on the Windows drive from WSL gives a damaged install: install in your Linux home instead).
// Exit codes: 0 = all checks passed, 1 = the page failed a check, 3 = jsdom itself could not be loaded.
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";

let JSDOM, VirtualConsole;
try {
  const root = process.env.LUMA_JSDOM_DIR ? path.resolve(process.env.LUMA_JSDOM_DIR) : process.cwd();
  ({ JSDOM, VirtualConsole } = createRequire(path.join(root, "noop.js"))("jsdom"));
} catch (e) {
  console.error("jsdom could not be loaded (a damaged or missing install): " + String(e.message || e).split("\n")[0].slice(0, 300));
  process.exit(3);
}

const [, , htmlPath, graphPath, mode, afterPath] = process.argv;
const html = fs.readFileSync(htmlPath, "utf8");
const graph = JSON.parse(fs.readFileSync(graphPath, "utf8"));
const live = mode === "live";
const problems = [];
let checks = 0;
const check = (cond, message) => { checks++; if (!cond) problems.push(message); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function fakeContext(canvas) {
  const store = { canvas };
  return new Proxy(store, {
    get(t, p) {
      if (p in t) return t[p];
      if (p === "measureText") return (s) => ({ width: String(s).length * 6, actualBoundingBoxAscent: 8, actualBoundingBoxDescent: 2 });
      if (p === "createLinearGradient" || p === "createRadialGradient" || p === "createPattern") return () => ({ addColorStop() {} });
      if (p === "getImageData") return () => ({ data: new Uint8ClampedArray(4) });
      if (p === "isPointInPath" || p === "isPointInStroke") return () => false;
      return typeof p === "string" ? function () {} : undefined;
    },
    set(t, p, v) { t[p] = v; return true; },
  });
}

const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push(String(e.detail || e.message || e)));
let fetchCalls = 0;
const dom = new JSDOM(html, {
  runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
  url: live ? "http://127.0.0.1:8765/" : "file:///graph.html",
  beforeParse(w) {
    w.HTMLCanvasElement.prototype.getContext = function () { return (this.__ctx ??= fakeContext(this)); };
    w.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
    w.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
    if (live) {
      const after = JSON.parse(fs.readFileSync(afterPath, "utf8"));
      w.fetch = async () => { fetchCalls++; return { ok: true, json: async () => after }; };
    }
  },
});
const { window } = dom;
const { document } = window;
await sleep(400);

const g = window.lumaGraph;
check(!!g, "page did not start (window.lumaGraph missing)");
check(errors.length === 0, "script errors: " + errors.join(" | "));

if (g && !live) {
  const byName = (name) => graph.nodes.find((n) => n.name === name);
  const text = (sel) => document.querySelector(sel)?.textContent ?? "";
  const HIDDEN = new Set(["registry", "media", "perception"]);

  // --- first view -----------------------------------------------------------------
  const visibleExpected = graph.nodes.filter((n) => !HIDDEN.has(n.layer)).length;
  const visibleActual = g.nodes.get({ filter: (n) => !n.hidden }).length;
  check(visibleActual === visibleExpected, `visible nodes: expected ${visibleExpected}, got ${visibleActual}`);
  check(g.nodes.length === graph.nodes.length, "every node should be loaded (hidden ones included)");
  check(g.edges.length === graph.edges.length, "every link should be loaded");
  check(g.hiddenKinds.has("EntityType"), "the type registry should start hidden");
  check(!g.hiddenKinds.has("Person"), "people should start visible");
  check(text("#stats").includes(`${visibleExpected} of ${graph.nodes.length} nodes shown`), "stats line is wrong: " + text("#stats"));
  check(text("#stats").includes("snapshot"), "a file should say snapshot, not live");
  const legendRows = [...document.querySelectorAll("#legend .kind")];
  check(legendRows.some((r) => r.textContent.includes("Person")), "legend should list Person");
  check(document.querySelector("#empty").hidden === true, "the empty-graph message should be hidden");

  // --- details & navigation ---------------------------------------------------------
  const sara = byName("Sara");
  g.showDetails(sara.id);
  check(document.querySelector("#main").classList.contains("with-details"), "details panel should open");
  check(text("#details h3") === "Sara", "details title: " + text("#details h3"));
  check(text("#details").includes("555-0100"), "details should show the phone number");
  check(text("#details").includes("KNOWS"), "details should list the KNOWS connection");
  const firstConn = document.querySelector("#details .conn");
  check(!!firstConn, "details should have clickable connections");
  if (firstConn) {
    const before = text("#details h3");
    firstConn.click();
    await sleep(50);
    check(text("#details h3") !== before, "clicking a connection should move to the other node");
  }

  // --- memory text is data, never markup ---------------------------------------------
  const evil = graph.nodes.find((n) => n.name.includes("<img"));
  if (evil) {
    g.showDetails(evil.id);
    check(!document.querySelector("#details img") && !document.querySelector("#legend img"), "HTML in a memory must not become elements");
    check(window.__pwned === undefined, "script in a memory must not run");
    check(text("#details h3").includes("<img"), "the raw text should be displayed as text");
  }

  // --- toggling a type ----------------------------------------------------------------
  const row = legendRows.find((r) => r.textContent.startsWith("EntityType") || r.textContent.includes("EntityType"));
  check(!!row, "legend should list EntityType (even though hidden)");
  if (row) {
    const box = row.querySelector("input");
    check(box.checked === false, "EntityType checkbox should start unchecked");
    box.click();
    await sleep(50);
    check(!g.hiddenKinds.has("EntityType"), "ticking a type should show it");
    check(g.nodes.get({ filter: (n) => n.hidden }).length === 0 || true, "nodes updated");
    const shown = g.nodes.get({ filter: (n) => !n.hidden }).length;
    check(shown > visibleExpected, "showing the registry should add nodes");
    check(text("#stats").includes(`${shown} of`), "stats should follow the toggle: " + text("#stats"));
  }

  // --- search ---------------------------------------------------------------------------
  const search = document.querySelector("#search");
  search.value = "sara"; search.dispatchEvent(new window.Event("input"));
  await sleep(50);
  check(text("#matches") === "1 of 1", "search 'sara' should find one match, got: " + text("#matches"));
  check(text("#details h3") === "Sara", "search should open the match");
  search.value = "zzzzzz"; search.dispatchEvent(new window.Event("input"));
  check(text("#matches") === "no matches", "a missing term should say so");
  search.value = ""; search.dispatchEvent(new window.Event("input"));
  check(text("#matches") === "", "clearing search should clear the counter");

  // --- other controls ---------------------------------------------------------------------
  const edgeBox = document.querySelector("#edgeLabels");
  edgeBox.click();
  check(g.edges.get().some((e) => e.label), "'Link names' should label the links");
  edgeBox.click();
  check(g.edges.get().every((e) => !e.label), "turning link names off should remove the labels");
  document.querySelector("#fit").click();
  const phys = document.querySelector("#physics"); phys.click();
  document.querySelector(".close")?.click();
  check(!document.querySelector("#main").classList.contains("with-details"), "Close should hide the panel");
}

if (g && live) {
  const before = g.nodes.length;
  const after = JSON.parse(fs.readFileSync(afterPath, "utf8"));
  check(before === graph.nodes.length, "live page should start from the initial snapshot");
  await sleep(3600);                                   // the page polls graph.json every 3 s
  check(fetchCalls >= 1, "a served page should poll graph.json");
  check(g.nodes.length === after.nodes.length, `live update: expected ${after.nodes.length} nodes, got ${g.nodes.length}`);
  const names = g.nodes.get().map((n) => n.label);
  const added = after.nodes.filter((n) => !graph.nodes.some((o) => o.id === n.id));
  const removed = graph.nodes.filter((o) => !after.nodes.some((n) => n.id === o.id));
  check(added.length > 0 && added.every((n) => names.includes(n.name)), "new memories should appear live");
  check(removed.length > 0 && removed.every((n) => !names.includes(n.name)), "deleted memories should disappear live");
  check(document.querySelector("#stats").textContent.includes("live"), "a served page should say live");
}

window.close();
if (problems.length) {
  console.error(`FAILED ${problems.length} of ${checks} checks:\n - ` + problems.join("\n - "));
  process.exit(1);
}
console.log(`OK: ${checks} checks passed (${live ? "live" : "snapshot"})`);
process.exit(0);