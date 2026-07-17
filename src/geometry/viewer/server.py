"""The viewer server: serves the browser UI and rebuilds the scene on demand.

Point it at a Python script that defines ``build() -> list`` and open the
browser page it prints::

    python -m geometry.viewer examples/viewer_demo.py --watch

Endpoints:
    GET  /            the UI (three.js single page)
    GET  /scene       latest scene JSON: {version, root, error, elapsed, script}
    GET  /status      {version, building}
    POST /rebuild     run the script again (in a subprocess) and swap the scene

``--watch`` polls the script file's mtime and rebuilds automatically on save,
so the page reflects the code in your IDE moments after you hit save (the page
polls /status and refetches when the version changes).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

REBUILD_TIMEOUT = 300.0  # seconds a build() may take before it is killed


class SceneStore:
    """The latest scene document plus a version counter (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._document: dict = {"root": None, "error": "No build yet.", "script": None}
        self.version = 0
        self.building = False

    def get(self) -> dict:
        """The current document plus its version number."""
        with self._lock:
            return {**self._document, "version": self.version}

    def set(self, document: dict) -> None:
        """Swap in a new document and bump the version."""
        with self._lock:
            self._document = document
            self.version += 1


class ViewerServer:
    """Own the scene store, the rebuild subprocess, and the optional file watcher."""

    def __init__(self, script: str | Path | None, port: int = 8735, watch: bool = False,
                 initial_document: dict | None = None):
        self.script = Path(script).resolve() if script is not None else None
        if self.script is not None and not self.script.exists():
            raise FileNotFoundError(self.script)
        self.port = port
        self.watch = watch and self.script is not None
        self.store = SceneStore()
        if initial_document is not None:
            self.store.set(initial_document)

    # -- rebuilding -------------------------------------------------------- #

    def rebuild(self) -> dict:
        """Run the user script in a fresh subprocess and swap in its scene."""
        if self.script is None:
            return {**self.store.get(),
                    "error": "rebuild unavailable: the scene did not come from a script file"}
        self.store.building = True
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                out_path = handle.name
            command = [sys.executable, "-m", "geometry.viewer._runner",
                       str(self.script), out_path]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=REBUILD_TIMEOUT,
                    check=False,  # non-zero exit handled below
                )
            except subprocess.TimeoutExpired:
                self.store.set({"root": None, "script": str(self.script),
                                "error": f"build() timed out after {REBUILD_TIMEOUT:.0f} s"})
                return self.store.get()
            try:
                document = json.loads(Path(out_path).read_text())
            except (OSError, json.JSONDecodeError):
                stderr = completed.stderr.strip()
                self.store.set({
                    "root": None, "script": str(self.script),
                    "error": f"runner crashed (exit {completed.returncode}):\n{stderr}",
                })
                return self.store.get()
            finally:
                Path(out_path).unlink(missing_ok=True)
            if completed.stderr.strip():
                print(completed.stderr.strip())  # surface the runner's progress log
            self.store.set(document)
            return self.store.get()
        finally:
            self.store.building = False

    def _watch_loop(self) -> None:
        last = self.script.stat().st_mtime
        while True:
            time.sleep(0.5)
            try:
                mtime = self.script.stat().st_mtime
            except OSError:
                continue
            if mtime != last:
                last = mtime
                self.rebuild()

    # -- serving ------------------------------------------------------------ #

    def serve_forever(self, open_browser: bool = True) -> None:
        """Start the HTTP server (blocking); initial build runs in the background."""
        if self.script is not None and self.store.version == 0:
            threading.Thread(target=self.rebuild, daemon=True).start()
        if self.watch:
            threading.Thread(target=self._watch_loop, daemon=True).start()

        server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler_class())
        url = f"http://127.0.0.1:{self.port}"
        print(f"geometry viewer: {url}  (script: {self.script})")
        if self.watch:
            print("watching for changes; the page refreshes on save")
        if open_browser:
            import webbrowser  # noqa: PLC0415

            threading.Timer(0.4, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nviewer stopped")

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # quiet
                pass

            def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: dict, status: int = 200) -> None:
                self._send(json.dumps(payload).encode(), "application/json", status)

            def do_GET(self):  # noqa: N802 (http.server API)
                if self.path in ("/", "/index.html"):
                    html = (resources.files("geometry.viewer") / "static" / "index.html")
                    self._send(html.read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/scene":
                    self._send_json(viewer.store.get())
                elif self.path == "/status":
                    self._send_json({"version": viewer.store.version,
                                     "building": viewer.store.building,
                                     "watch": viewer.watch})
                else:
                    self._send_json({"error": "not found"}, status=404)

            def do_POST(self):  # noqa: N802 (http.server API)
                if self.path == "/rebuild":
                    document = viewer.rebuild()
                    self._send_json({"version": document["version"],
                                     "error": document.get("error")})
                else:
                    self._send_json({"error": "not found"}, status=404)

        return Handler
