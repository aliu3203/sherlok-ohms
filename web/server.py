"""Tiny web frontend for main.py: upload a schema, start/stop the checker, stream its logs.
Stdlib only, so it runs on the Pi (QNX) with nothing extra installed.

  python3 web/server.py                       # on the Pi, from the repo root
  open http://<pi-ip>:8000                    # or open web/index.html on the laptop and enter the Pi's address

  POST /run     {"name": "x.txt", "text": "..."}  or  {"schema": "schemas/demo.txt"}
  POST /stop
  GET  /schemas  -> ["schemas/demo.txt", ...]
  GET  /status   -> {"running": bool, "schema": str|null}
  GET  /logs     -> text/event-stream of main.py output (replays the current run first)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "schemas" / "uploads"
INDEX = Path(__file__).resolve().parent / "index.html"
MAX_LINES = 5000


class Runner:
    """One main.py process at a time; its output lines are kept with sequence numbers for SSE replay."""

    def __init__(self, extra_args: list[str]):
        self.extra_args = extra_args
        self.proc: subprocess.Popen | None = None
        self.pump: threading.Thread | None = None
        self.schema: str | None = None
        self.lines: list[tuple[int, str]] = []
        self.seq = 0
        self.cond = threading.Condition()

    def _emit(self, line: str):
        with self.cond:
            self.seq += 1
            self.lines.append((self.seq, line))
            del self.lines[:-MAX_LINES]
            self.cond.notify_all()

    def start(self, schema: str):
        self.stop()
        cmd = [sys.executable, "-u", "main.py", "--schema", schema, *self.extra_args]
        with self.cond:
            self.lines.clear()
        self.schema = schema
        self._emit(f"$ {' '.join(cmd[1:])}")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.pump = threading.Thread(target=self._pump, args=(self.proc,), daemon=True)
        self.pump.start()

    def _pump(self, proc: subprocess.Popen):
        for line in proc.stdout:
            self._emit(line.rstrip("\n"))
        self._emit(f"[exited with code {proc.wait()}]")

    def stop(self):
        p = self.proc
        if p is None or p.poll() is not None:
            return
        p.send_signal(signal.SIGINT)  # main.py catches KeyboardInterrupt and turns the LEDs off
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        self.pump.join(2)  # let its "[exited ...]" line land before a new run clears the log

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def save_upload(name: str, text: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(name or "upload.txt"))
    if ext.lower() not in (".txt", ".json"):
        raise ValueError("schema must be .txt or .json")
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem) or "upload"
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"{stem}{ext.lower()}"
    path.write_text(text)
    return str(path.relative_to(ROOT))


class Handler(BaseHTTPRequestHandler):
    runner: Runner

    def log_message(self, fmt, *args):  # keep the Pi console quiet
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        r = self.runner
        if self.path in ("/", "/index.html"):
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/schemas":
            self._json(sorted(str(p.relative_to(ROOT)) for ext in ("*.txt", "*.json")
                              for p in (ROOT / "schemas").rglob(ext)))
        elif self.path == "/status":
            self._json({"running": r.running(), "schema": r.schema})
        elif self.path.startswith("/logs"):
            self._stream(int(self.headers.get("Last-Event-ID") or 0))
        else:
            self._json({"error": "not found"}, 404)

    def _stream(self, last: int):
        r = self.runner
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                with r.cond:
                    if last > r.seq:
                        last = 0  # browser reconnected after a server restart
                    if r.lines and r.lines[0][0] > last + 1:
                        last = r.lines[0][0] - 1  # new run (or trimmed): replay from its first line
                    new = [(s, l) for s, l in r.lines if s > last]
                    if not new:
                        r.cond.wait(15)
                        new = [(s, l) for s, l in r.lines if s > last]
                chunk = "".join(f"id: {s}\ndata: {l}\n\n" for s, l in new) or ": ping\n\n"
                self.wfile.write(chunk.encode())
                self.wfile.flush()
                if new:
                    last = new[-1][0]
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        r = self.runner
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/run":
                if "text" in req:
                    schema = save_upload(req.get("name", ""), req["text"])
                else:
                    schema = req.get("schema", "")
                    if not (ROOT / schema).resolve().is_relative_to(ROOT / "schemas") or not (ROOT / schema).is_file():
                        raise ValueError(f"no such schema: {schema}")
                r.start(schema)
                self._json({"running": True, "schema": schema})
            elif self.path == "/stop":
                r.stop()
                self._json({"running": False, "schema": r.schema})
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description="Web frontend for main.py. Unknown args are passed to main.py.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a, rest = ap.parse_known_args()
    if "--source" not in rest:
        rest += ["--source", "qnx"]
    if "--model" not in rest:
        rest += ["--model", "models/holes.tflite"]
    Handler.runner = runner = Runner(rest)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"sherlok web on http://{a.host}:{a.port}  (main.py args: {' '.join(rest)})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
