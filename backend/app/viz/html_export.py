"""Turn a graph snapshot into ONE self-contained HTML page (the graph library is inlined: no network)."""
import json
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
VIS_MARKER = "/*__VIS_JS__*/"
DATA_MARKER = "/*__GRAPH_DATA__*/"


def json_for_script(data) -> str:
    """JSON that is safe inside a <script> block: '<' becomes \\u003c so memory text can never
    contain '</script>' or '<!--', and the two line separators JS treats as newlines are escaped."""
    return (json.dumps(data, ensure_ascii=False)
            .replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render_html(graph: dict) -> str:
    template = (ASSETS / "template.html").read_text(encoding="utf-8")
    vis = (ASSETS / "vis-network.min.js").read_text(encoding="utf-8")
    # Split once, on the template only, so neither the library nor the data can collide with a marker.
    head, _, rest = template.partition(VIS_MARKER)
    middle, _, tail = rest.partition(DATA_MARKER)
    return head + vis + middle + json_for_script(graph) + tail