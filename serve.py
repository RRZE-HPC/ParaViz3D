#!/usr/bin/env python3
"""Static file server that answers HTTP Range requests.

`python3 -m http.server` ignores the `Range` header entirely — it replies
`200 OK` with the whole file — which defeats the point of the PV3D format,
where a viewer fetches one block at a time out of a multi-gigabyte file.
This server answers `206 Partial Content` so the browser really does transfer
only the requested bytes.

    python3 serve.py [--port 8000] [--directory .]

The format also works with no server at all: with a local file chosen through
an <input type="file">, `file.slice(a, b).arrayBuffer()` reads just that byte
range from disk.
"""

from __future__ import annotations

import argparse
import http.server
import io
import json
import os
import re
import socketserver

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)\s*$")


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus single-range support."""

    def end_headers(self) -> None:
        # Advertise range support and allow the viewer to be served from
        # somewhere else during development.
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length")
        super().end_headers()

    def send_head(self):
        if self.path.split("?", 1)[0].rstrip("/") == "/api/list":
            return self._send_listing()

        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()

        m = _RANGE_RE.match(rng)
        if not m:
            self.send_error(400, "malformed Range header")
            return None

        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            fh = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(fh.fileno()).st_size
        first, last = m.group(1), m.group(2)
        if first == "":
            # "bytes=-N" means the final N bytes.
            length = int(last or 0)
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
        end = min(end, size - 1)

        if start > end or start >= size:
            fh.close()
            self.send_response(416, "Requested Range Not Satisfiable")
            self.send_header("Content-Range", "bytes */%d" % size)
            self.end_headers()
            return None

        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        fh.seek(start)
        return _BoundedReader(fh, end - start + 1)


    #: Where trace files live, relative to the served directory.  `api/files` holds the
    #: legacy text traces, `api/pv3d` the curated converted ones, and `traces` whatever the
    #: converter wrote next to an OTF2 folder.
    LISTING_ROOTS = ("api/files", "api/pv3d", "traces")
    LISTING_SUFFIXES = (".txt", ".pv3dbin", ".pv3dtxt")

    def _send_listing(self):
        """GET /api/list -> [{path, bytes, format}], sorted by path.

        api/files/list.txtt is hand-maintained and has drifted out of date; walking the
        directories cannot.  The viewer falls back to the static file when this is missing,
        so opening the page from any other static server still works.
        """
        root = os.path.abspath(self.directory)
        entries = []
        for rel_root in self.LISTING_ROOTS:
            base = os.path.join(root, rel_root)
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, files in os.walk(base):
                for name in files:
                    if not name.endswith(self.LISTING_SUFFIXES):
                        continue
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    entries.append({
                        "path": rel,
                        "bytes": os.path.getsize(full),
                        "format": name.rsplit(".", 1)[-1],
                    })
        entries.sort(key=lambda e: e["path"])
        body = json.dumps(entries).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        return io.BytesIO(body)


class _BoundedReader:
    """File wrapper that yields at most `remaining` bytes, for copyfile()."""

    def __init__(self, fh, remaining: int):
        self._fh = fh
        self._remaining = remaining

    def read(self, amount: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if amount is None or amount < 0:
            amount = self._remaining
        data = self._fh.read(min(amount, self._remaining))
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        self._fh.close()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--directory", default=".")
    args = ap.parse_args()

    handler = lambda *a, **kw: RangeRequestHandler(*a, directory=args.directory, **kw)
    with _Server((args.bind, args.port), handler) as httpd:
        print("serving %s at http://%s:%d/ (Range requests supported)"
              % (os.path.abspath(args.directory), args.bind, args.port))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
