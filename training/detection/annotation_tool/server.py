#!/usr/bin/env python3
"""training/detection/annotation_tool/server.py — Local-only web server
for the human bounding-box annotation tool.

Run it, then open the printed URL in a browser:

    python training/detection/annotation_tool/server.py \
        --images-dir "E:/SolarAI_Datasets/domain_adaptation_v1/annotation/images" \
        --labels-dir "E:/SolarAI_Datasets/domain_adaptation_v1/annotation/labels"

Binds to 127.0.0.1 only - never exposed beyond this machine. Uses only
the Python standard library (no new dependency). All box save/load
logic lives in core.py and is unit-tested there; this file is a thin,
mostly-untested-by-design HTTP wrapper (path routing, JSON
(de)serialization, path-traversal guards).
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from training.detection.annotation_tool.core import (
    compute_progress,
    list_images,
    load_boxes,
    save_boxes,
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _safe_filename(raw: str, allowed_names: set[str]) -> str | None:
    """Only ever resolves to a name that already exists in the given
    directory's own listing - blocks path traversal (`../`, absolute
    paths, etc.) by construction rather than by pattern-blacklisting."""
    name = Path(unquote(raw)).name
    return name if name in allowed_names else None


class Handler(BaseHTTPRequestHandler):
    images_dir: Path
    labels_dir: Path

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write(f"[annotation-tool] {self.address_string()} - {format % args}\n")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path == "/" or path == "/index.html":
                self._send_file(_STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif path.startswith("/static/"):
                rel = path[len("/static/"):]
                target = (_STATIC_DIR / rel).resolve()
                if _STATIC_DIR.resolve() not in target.parents and target != _STATIC_DIR.resolve():
                    self._send_error_json(403, "forbidden")
                    return
                if not target.is_file():
                    self._send_error_json(404, "not found")
                    return
                self._send_file(target)
            elif path.startswith("/images/"):
                allowed = set(list_images(self.images_dir))
                name = _safe_filename(path[len("/images/"):], allowed)
                if name is None:
                    self._send_error_json(404, "image not found")
                    return
                self._send_file(self.images_dir / name)
            elif path == "/api/progress":
                self._send_json(compute_progress(self.images_dir, self.labels_dir))
            elif path == "/api/boxes":
                qs = dict(p.split("=", 1) for p in (parsed.query.split("&") if parsed.query else []))
                allowed = set(list_images(self.images_dir))
                name = _safe_filename(qs.get("filename", ""), allowed)
                if name is None:
                    self._send_error_json(404, "image not found")
                    return
                boxes = load_boxes(self.labels_dir, name)
                self._send_json({"filename": name, "boxes": boxes})
            else:
                self._send_error_json(404, "not found")
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/boxes":
            self._send_error_json(404, "not found")
            return
        try:
            qs = dict(p.split("=", 1) for p in (parsed.query.split("&") if parsed.query else []))
            allowed = set(list_images(self.images_dir))
            name = _safe_filename(qs.get("filename", ""), allowed)
            if name is None:
                self._send_error_json(404, "image not found")
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8"))
            boxes = payload.get("boxes", [])

            save_boxes(self.labels_dir, name, boxes)
            self._send_json({"ok": True, "filename": name, "saved_box_count": len(boxes)})
        except ValueError as exc:
            # A rejected box (validate_box) - a real, expected client
            # error, not a server bug - reported as 400, not 500.
            self._send_error_json(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")


def make_handler(images_dir: Path, labels_dir: Path) -> type[Handler]:
    class BoundHandler(Handler):
        pass

    BoundHandler.images_dir = images_dir
    BoundHandler.labels_dir = labels_dir
    return BoundHandler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        print(f"error: images dir does not exist: {args.images_dir}", file=sys.stderr)
        return 1
    args.labels_dir.mkdir(parents=True, exist_ok=True)

    handler_cls = make_handler(args.images_dir.resolve(), args.labels_dir.resolve())
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), handler_cls)
    n_images = len(list_images(args.images_dir))
    print(f"Serving {n_images} images from {args.images_dir}")
    print(f"Labels will be saved to {args.labels_dir}")
    print(f"Open: http://127.0.0.1:{args.port}/  (local only - not exposed to the network)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
