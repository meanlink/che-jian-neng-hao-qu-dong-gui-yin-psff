"""协调服务：HTTP API + 演练页面。

纯标准库实现（ThreadingHTTPServer）。所有状态变更经 Store 持锁落盘，
服务中断后用同一数据目录重启即可恢复已确认阶段。
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .store import Store, StoreError

PAGE_PATH = Path(__file__).with_name("page.html")


def _read_body(handler):
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise StoreError(400, "bad_json", "请求体必须为JSON") from None


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FedCoordinator/1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # 静默访问日志，演练输出由事件流承担
            pass

        # ---------------------------------------------------------- 工具
        def _send(self, status, obj, content_type="application/json; charset=utf-8"):
            if isinstance(obj, (dict, list)):
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            elif isinstance(obj, str):
                body = obj.encode("utf-8")
            else:
                body = obj
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, exc):
            self._send(exc.status, {"error": {"code": exc.code, "message": exc.message}})

        def _query(self):
            from urllib.parse import urlparse, parse_qs
            parts = urlparse(self.path)
            return parts.path, {k: v[0] for k, v in parse_qs(parts.query).items()}

        # ---------------------------------------------------------- GET
        def do_GET(self):
            path, query = self._query()
            try:
                if path == "/":
                    self._send(200, PAGE_PATH.read_text(encoding="utf-8"),
                               content_type="text/html; charset=utf-8")
                elif path == "/api/health":
                    self._send(200, {"ok": True})
                elif path == "/api/overview":
                    self._send(200, {"runs": store.list_runs(), "sites": store.list_sites()})
                elif path.startswith("/api/runs/"):
                    self._get_run(path, query)
                elif path == "/api/sites":
                    self._send(200, {"sites": store.list_sites()})
                else:
                    self._send(404, {"error": {"code": "not_found", "message": "未知路径"}})
            except StoreError as exc:
                self._error(exc)

        def _get_run(self, path, query):
            rest = path[len("/api/runs/"):]
            if rest.endswith("/brief"):
                run_id = rest[: -len("/brief")].rstrip("/")
                site_id = query.get("site_id", "")
                self._send(200, store.brief(run_id, site_id))
            elif rest.endswith("/result"):
                run_id = rest[: -len("/result")].rstrip("/")
                self._send(200, store.result(run_id))
            else:
                self._send(200, store.run_detail(rest))

        # ---------------------------------------------------------- POST
        def do_POST(self):
            path, _ = self._query()
            try:
                body = _read_body(self)
                if path == "/api/runs":
                    run = store.create_run(body.get("params"), body.get("boundary"),
                                           body.get("participants"), body.get("created_by", ""))
                    self._send(201, {"run_id": run["run_id"],
                                     "manifest_hash": run["manifest"]["manifest_hash"]})
                elif path == "/api/sites/register":
                    entry = store.register_site(str(body.get("site_id", "")),
                                                str(body.get("control_url", "")),
                                                str(body.get("sample_version", "")),
                                                int(body.get("records", 0)),
                                                body.get("coverage"))
                    self._send(200, entry)
                elif path == "/api/admin/recover":
                    self._send(200, {"recovered": store.recover_all()})
                elif path.startswith("/api/runs/"):
                    self._post_run(path, body)
                elif path.startswith("/api/sites/") and path.endswith("/drill/disconnect"):
                    site_id = path[len("/api/sites/"): -len("/drill/disconnect")].rstrip("/")
                    self._proxy_disconnect(site_id, body)
                elif path.startswith("/api/sites/") and path.endswith("/drill/resume"):
                    site_id = path[len("/api/sites/"): -len("/drill/resume")].rstrip("/")
                    self._proxy_control(site_id, "resume")
                else:
                    self._send(404, {"error": {"code": "not_found", "message": "未知路径"}})
            except StoreError as exc:
                self._error(exc)

        def _post_run(self, path, body):
            rest = path[len("/api/runs/"):]
            if rest.endswith("/contributions"):
                run_id = rest[: -len("/contributions")].rstrip("/")
                env = dict(body)
                env["run_id"] = run_id
                self._send(200, store.submit(env))
            elif rest.endswith("/approve"):
                run_id = rest[: -len("/approve")].rstrip("/")
                run = store.approve(run_id, body.get("approver", ""), body.get("note", ""))
                self._send(200, {"run_id": run_id, "status": run["status"], "approval": run["approval"]})
            elif rest.endswith("/resume"):
                run_id = rest[: -len("/resume")].rstrip("/")
                self._send(200, store.resume(run_id))
            elif rest.endswith("/fail"):
                run_id = rest[: -len("/fail")].rstrip("/")
                run = store.fail(run_id, str(body.get("site_id", "")), str(body.get("reason", "")))
                self._send(200, {"run_id": run_id, "status": run["status"], "failure": run["failure"]})
            elif "/drill/" in rest:
                run_id, mode = rest.split("/drill/")
                outcome = store.drill_replay(run_id, str(body.get("stage", "")),
                                             str(body.get("site_id", "")), mode)
                self._send(200, {"drill": mode, "outcome": outcome})
            else:
                self._send(404, {"error": {"code": "not_found", "message": "未知路径"}})

        def _proxy_disconnect(self, site_id, body):
            site = store.sites.get(site_id)
            if not site or not site.get("control_url"):
                raise StoreError(404, "site_offline", f"站点{site_id}未注册控制地址，可能已断线")
            seconds = int(body.get("seconds", 30))
            try:
                req = urllib.request.Request(
                    site["control_url"].rstrip("/") + "/control/disconnect",
                    data=json.dumps({"seconds": seconds}).encode("utf-8"),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self._send(200, {"site_id": site_id, "drill": "disconnect", "result": payload})
            except OSError as exc:
                raise StoreError(502, "site_unreachable", f"站点{site_id}控制端不可达：{exc}") from None

        def _proxy_control(self, site_id, action):
            site = store.sites.get(site_id)
            if not site or not site.get("control_url"):
                raise StoreError(404, "site_offline", f"站点{site_id}未注册控制地址，可能已断线")
            try:
                req = urllib.request.Request(
                    site["control_url"].rstrip("/") + f"/control/{action}",
                    data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self._send(200, {"site_id": site_id, "result": payload})
            except OSError as exc:
                raise StoreError(502, "site_unreachable", f"站点{site_id}控制端不可达：{exc}") from None

    return Handler


def build_server(store: Store, host="127.0.0.1", port=8310):
    return ThreadingHTTPServer((host, port), make_handler(store))


def main():
    parser = argparse.ArgumentParser(description="集团联合能耗归因协调服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8310)
    parser.add_argument("--data-dir", required=True, help="运行状态数据目录（中断恢复用）")
    args = parser.parse_args()
    store = Store(args.data_dir)
    recovered = store.recover_all()
    if recovered:
        print(f"[coordinator] 恢复扫描推进 {len(recovered)} 个运行")
    server = build_server(store, args.host, args.port)
    print(f"[coordinator] 页面与API: http://{args.host}:{args.port}/  数据目录: {args.data_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
