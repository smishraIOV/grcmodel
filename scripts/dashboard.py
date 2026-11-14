"""Serve the what-if dashboard on this machine (docs/plan-tool.md step 10).

    uv run python -m scripts.dashboard            then open http://127.0.0.1:8765
    uv run python -m scripts.dashboard --port 9000 --no-browser

One page (`dashboard/index.html`) and two addresses behind it: `/inputs`
lists every input with the example firm's values, and `/api` answers a
request as `quant.api.respond` does. The server listens on this machine only.
"""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from quant.api import inputs, respond
from quant.cli import profile_parser
from quant.numerics import NumericsProfile, get_profile

PAGE = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
MAX_REQUEST = 1_000_000     # bytes; a request is a few kilobytes


def make_server(port: int, numerics: NumericsProfile, page: Path = PAGE) -> ThreadingHTTPServer:
    """The dashboard's server, bound to this machine only. Port 0 picks a free one."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, kind: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(data).encode())

        def do_GET(self) -> None:      # noqa: N802 (the base class's name)
            where = urlsplit(self.path).path          # the page takes options after a '?'
            if where in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", page.read_bytes())
            elif where == "/inputs":
                self._json(200, inputs())
            else:
                self._json(404, {"ok": False, "problems": [f"there is nothing at {self.path}"]})

        def do_POST(self) -> None:     # noqa: N802
            if urlsplit(self.path).path != "/api":
                self._json(404, {"ok": False, "problems": [f"there is nothing at {self.path}"]})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_REQUEST:
                self._json(400, {"ok": False, "problems": ["the request is empty or too large"]})
                return
            try:
                request = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as error:
                self._json(400, {"ok": False, "problems": [f"the request could not be read: {error}"]})
                return
            self._json(200, respond(request, numerics))

        def log_message(self, *args) -> None:
            pass                        # quiet: one line per request helps nobody

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = profile_parser(__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page in a browser")
    args = parser.parse_args(argv)
    server = make_server(args.port, get_profile(args.profile))
    address = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"The dashboard is at {address}  (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [address]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
