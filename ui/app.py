"""Local evaluator client. Run: python ui/app.py --api http://SERVER:8000"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

STATIC = Path(__file__).resolve().parent / "static"

class Client(SimpleHTTPRequestHandler):
    def forward(self, body=None):
        endpoint = self.path.removeprefix("/api")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        request = Request(self.server.backend + endpoint, data=body, headers=headers)
        try:
            with urlopen(request, timeout=90) as response:
                status, result = response.status, response.read()
        except HTTPError as exc:
            status, result = exc.code, exc.read()
        except (URLError, TimeoutError, OSError):
            status = 502
            result = b'{"detail":"Laptop 1 is unreachable. Check the server address and Wi-Fi connection."}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)

    def do_GET(self):
        if self.path in ("/api/health", "/api/trips"):
            self.forward()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path != "/api/predict":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2048:
                raise ValueError
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or not isinstance(data.get("trip_id"), str):
                raise ValueError
        except (ValueError, UnicodeError):
            return self.send_error(400, "Expected a JSON trip_id string")
        self.forward(json.dumps({"trip_id": data["trip_id"]}).encode())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.environ.get("PATHSNAP_API_URL"))
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not args.api:
        parser.error("Set --api to Laptop 1's LAN URL, for example http://192.168.1.23:8000")
    url = urlsplit(args.api)
    if url.scheme not in ("http", "https") or not url.hostname or url.path not in ("", "/") or url.query or url.fragment:
        parser.error("Use an HTTP(S) base address without an endpoint")
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Client, directory=str(STATIC))) as server:
        server.backend = args.api.rstrip("/")
        print(f"Open http://127.0.0.1:{args.port} on Laptop 2. Backend: {server.backend}", flush=True)
        server.serve_forever()

if __name__ == "__main__":
    main()
