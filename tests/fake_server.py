"""A fake OpenAI-compatible streaming server with a known cost model, for offline tests.

Each request waits `ttft` seconds plus a queueing delay that grows with requests in flight, then
streams `max_tokens` tokens at `tpot` seconds each, then a usage chunk. It also serves /health and
vLLM-style /metrics (TTFT histogram, prefix-cache counters) so the second measurement can be checked.
Quality prompts ("What is a * b + c?") get the right answer back.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class State:
    def __init__(self, ttft=0.05, tpot=0.002, queue_per_inflight=0.01, usage=True, key="test-key"):
        self.ttft, self.tpot, self.q, self.usage, self.key = ttft, tpot, queue_per_inflight, usage, key
        self.inflight = 0
        self.lock = threading.Lock()
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.prefix_queries = 0
        self.prefix_hits = 0
        self.seen_prefixes: set[str] = set()
        self.bodies: list[dict] = []


def make_handler(st: State):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _plain(self, code: int, body: bytes = b""):
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._plain(200, b"ok")
            if self.path == "/metrics":
                body = (f'vllm:time_to_first_token_seconds_sum{{model_name="m"}} {st.ttft_sum}\n'
                        f'vllm:time_to_first_token_seconds_count{{model_name="m"}} {st.ttft_count}\n'
                        f'vllm:time_to_first_token_seconds_bucket{{le="0.1",model_name="m"}} 1\n'
                        f'vllm:prefix_cache_queries_total{{model_name="m"}} {st.prefix_queries}\n'
                        f'vllm:prefix_cache_hits_total{{model_name="m"}} {st.prefix_hits}\n').encode()
                return self._plain(200, body)
            if self.path == "/v1/models":
                return self._plain(200, json.dumps({"data": [{"id": "fake/model"}]}).encode())
            return self._plain(404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            st.bodies.append(req)
            if self.headers.get("Authorization") != f"Bearer {st.key}":
                return self._plain(401)
            content = req["messages"][0]["content"]
            prefix = content[:2000]
            with st.lock:
                st.inflight += 1
                wait = st.ttft + st.q * (st.inflight - 1)
                st.prefix_queries += 1
                if prefix in st.seen_prefixes:
                    st.prefix_hits += 1
                    wait *= 0.3
                st.seen_prefixes.add(prefix)
            t0 = time.perf_counter()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(obj):
                data = f"data: {obj if isinstance(obj, str) else json.dumps(obj)}\n\n".encode()
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            time.sleep(wait)
            with st.lock:
                st.ttft_sum += time.perf_counter() - t0
                st.ttft_count += 1
            m = re.match(r"What is (\d+) \* (\d+) \+ (\d+)\?", content)
            tokens = [str(int(m[1]) * int(m[2]) + int(m[3]))] if m else ["x"] * req["max_tokens"]
            for i, tok in enumerate(tokens):
                last = i == len(tokens) - 1
                send({"choices": [{"index": 0, "delta": {"content": tok},
                                   "finish_reason": ("stop" if m else "length") if last else None}]})
                time.sleep(st.tpot)
            if st.usage:
                prompt_tokens = int(len(content.split()) * 1.3) + 60
                send({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(tokens)}})
            send("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            with st.lock:
                st.inflight -= 1

    return H


def start(st: State) -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(st))
    srv.handle_error = lambda *a: None   # clients closing keep-alive sockets is normal here
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
