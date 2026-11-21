"""시연 화면 정적 서버. 마이크 권한 때문에 반드시 http://localhost 로 연다.

    python demo/serve.py            # → http://localhost:8080
"""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    handler = functools.partial(NoCache, directory=str(Path(__file__).resolve().parent))
    print(f"READY http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()
