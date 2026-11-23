#!/usr/bin/env python3
"""轻量 Web 服务：仅用 Python 标准库，零额外依赖。

提供静态前端页面与 Agent 调用接口，供 start.bat 一键启动。
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
sys.path.insert(0, str(BASE_DIR))

from app.graph.workflow import run_agent  # noqa: E402
from app.models.database import init_db  # noqa: E402
from app.tools.registry import tool_names  # noqa: E402

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            file_path = WEB_DIR / filename
            if not file_path.exists():
                self._send_json(500, {"error": f"缺少前端文件 {filename}"})
                return
            self._send(200, file_path.read_bytes(), content_type)
        elif path == "/api/health":
            self._send_json(200, {"status": "ok", "tools": tool_names()})
        elif path == "/api/tools":
            self._send_json(200, {"tools": tool_names()})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/chat":
            self._send_json(404, {"error": "Not Found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            message = str(payload.get("message", "")).strip()
            history = payload.get("history") or []
            if not isinstance(history, list):
                history = []
            decision = payload.get("decision")
            thread_id = payload.get("thread_id")
            if decision is None and not message:
                self._send_json(400, {"error": "消息不能为空"})
                return
            result = run_agent(
                message,
                history=history,
                thread_id=str(thread_id) if thread_id else None,
                decision=decision,
            )
            self._send_json(200, result)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                500,
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    def log_message(self, fmt: str, *args) -> None:
        print(f"[HTTP] {fmt % args}")


def pick_port(start: int = 8765, attempts: int = 50) -> int:
    """从 start 起找一个可用端口，避免与已运行实例冲突。"""
    for port in range(start, start + attempts):
        sock = socket.socket()
        try:
            sock.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    raise RuntimeError(f"端口 {start}-{start + attempts - 1} 均被占用")


def main() -> None:
    init_db()
    port = pick_port()
    url = f"http://127.0.0.1:{port}/"

    def open_browser() -> None:
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("=" * 60)
    print("  日程任务自动编排 Agent 已启动")
    print(f"  浏览器地址: {url}")
    print("  关闭窗口即可停止服务")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
