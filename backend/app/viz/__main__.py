"""Visualise Luma's memory graph.  Run from the backend/ folder:

    python -m backend.app.viz                  write data/graph.html (a snapshot) and print where it is
    python -m backend.app.viz --open           ...and open it in your browser
    python -m backend.app.viz --serve          live view at http://127.0.0.1:8765, refreshing as Luma learns

It reads the database READ-ONLY, so it is safe to run while `python -m app.cli` is running.
The page contains your memories in plain text: don't commit or share it.
"""
import argparse
import json
import platform
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from backend.app.config import settings
from backend.app.viz.graph_export import read_graph
from backend.app.viz.html_export import render_html


def windows_path(path: Path) -> str | None:
    """Under WSL, the Windows spelling of a path (so you can open it from Explorer)."""
    if "microsoft" not in platform.release().lower():
        return None
    try:
        out = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - purely a convenience
        return None


def make_handler(db_path: str, max_nodes: int):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            port = self.server.server_port
            if self.headers.get("Host", "") not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
                return self._send(403, b"Forbidden", "text/plain")  # blocks DNS-rebinding style access
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, render_html(read_graph(db_path, max_nodes)).encode(), "text/html; charset=utf-8")
                if path == "/graph.json":
                    return self._send(200, json.dumps(read_graph(db_path, max_nodes)).encode(), "application/json")
            except Exception as e:  # noqa: BLE001 - report instead of dropping the connection
                return self._send(500, f"Could not read the database: {e}".encode(), "text/plain")
            self._send(404, b"Not found", "text/plain")

        def log_message(self, *args):  # keep the terminal quiet
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.viz", description="Visualise Luma's memory graph.")
    ap.add_argument("--db", default=settings.db_path, help="database path (default: LUMA_DB_PATH)")
    ap.add_argument("--out", help="where to write the HTML snapshot (default: next to the database)")
    ap.add_argument("--open", action="store_true", help="open the snapshot in your browser")
    ap.add_argument("--serve", action="store_true", help="serve a live view on 127.0.0.1 instead of writing a file")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-nodes", type=int, default=3000, help="cap for very large graphs")
    args = ap.parse_args(argv)

    db = Path(args.db)
    if not db.exists():
        print(f"No database at {db.resolve()}. Run `python -m app.cli` once and tell Luma something first,\n"
              f"or point to your database with --db.", file=sys.stderr)
        return 1

    if args.serve:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(str(db), args.max_nodes))
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Live graph at {url}   (Ctrl+C to stop; only reachable from this computer)")
        if args.open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            server.server_close()
        return 0

    graph = read_graph(str(db), args.max_nodes)
    out = Path(args.out) if args.out else db.resolve().parent / "graph.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(graph), encoding="utf-8")
    s = graph["stats"]
    print(f"Wrote {out.resolve()}  ({s['nodes']} nodes, {s['edges']} links)")
    win = windows_path(out.resolve())
    if win:
        print(f"Open it from Windows: {win}")
    print("It contains your memories in plain text: don't commit or share it.")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())