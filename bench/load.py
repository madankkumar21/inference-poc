"""Load against any OpenAI-compatible chat endpoint, with streaming timings.

Two load shapes:
  closed loop  `c` workers, each sends its next request when the last one finishes, for a fixed time.
               The first request of each worker (the first wave) is marked warmup: it pays the TLS
               handshake and arrives as a burst, so the report leaves it out.
  open loop    requests arrive as a Poisson process at a fixed rate, whatever the server is doing.
               This is how real traffic arrives, and it shows queueing that a closed loop hides.

TTFT is measured on this client, so it includes the network round trip; rtt() measures that round
trip separately. For vLLM, read_metrics() gives the server's own TTFT histogram as a second measurement.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import statistics
import time
from typing import Callable

import httpx

from providers.base import Endpoint

TIMEOUT = httpx.Timeout(connect=30, read=600, write=60, pool=600)
# One request may not take longer than this from send to last token. A request that does is an error,
# not a sample. No self-hosted request came near it (slowest: 28 s); Together stalled a few for 53-145 s,
# and a closed-loop level waits for its last worker, so one stall stretched the whole level.
REQUEST_DEADLINE_S = 60.0


def _itl_stats(chunk_times: list[float]) -> tuple[float | None, float | None, float | None]:
    gaps = [b - a for a, b in zip(chunk_times, chunk_times[1:])]
    if len(gaps) < 2:
        return None, None, None
    gaps.sort()
    p99 = gaps[min(len(gaps) - 1, int(round(0.99 * (len(gaps) - 1))))]
    return statistics.median(gaps), p99, gaps[-1]


async def one_request(client: httpx.AsyncClient, ep: Endpoint, prompt: str, max_tokens: int, tag: dict) -> dict:
    body = {
        "model": ep.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **ep.extra_body,
    }
    if ep.honours_min_tokens and tag.get("workload") != "quality":
        body["min_tokens"] = max_tokens
    row = {**tag, "provider": ep.provider, "variant": ep.variant, "model": ep.model,
           "max_tokens": max_tokens, "sent_at": time.time()}
    t0 = time.perf_counter()
    ttft = None
    usage = None
    finish = None
    chunk_times: list[float] = []
    text: list[str] = []
    err = None
    try:
        async with asyncio.timeout(REQUEST_DEADLINE_S):
            async with client.stream("POST", f"{ep.base_url}/chat/completions", json=body,
                                     headers=ep.headers, timeout=TIMEOUT) as r:
                if r.status_code != 200:
                    err = f"http {r.status_code}: {(await r.aread())[:300]!r}"
                else:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        obj = json.loads(data)
                        if obj.get("error"):
                            err = f"stream error: {json.dumps(obj['error'])[:300]}"
                            break
                        if obj.get("usage"):
                            usage = obj["usage"]
                        for ch in obj.get("choices") or []:
                            delta = ch.get("delta") or {}
                            if delta.get("content"):
                                now = time.perf_counter()
                                if ttft is None:
                                    ttft = now - t0
                                chunk_times.append(now)
                                text.append(delta["content"])
                            if ch.get("finish_reason"):
                                finish = ch["finish_reason"]
    except TimeoutError:
        err = f"deadline: no complete response in {REQUEST_DEADLINE_S:.0f} s"
    except Exception as e:   # recorded, not raised: an error rate is a result
        err = repr(e)
    e2e = time.perf_counter() - t0
    if err is None and ttft is None:
        err = "no content tokens"
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens")
    usage_source = "usage"
    if ct is None and chunk_times:
        ct, usage_source = len(chunk_times), "chunk_count"   # a lower bound when chunks carry several tokens
    tpot = (e2e - ttft) / (ct - 1) if (ttft is not None and ct and ct > 1) else None
    itl_p50, itl_p99, itl_max = _itl_stats(chunk_times)
    row.update({"ttft_s": ttft, "e2e_s": e2e, "prompt_tokens": pt, "completion_tokens": ct,
                "usage_source": usage_source, "finish_reason": finish, "content_chunks": len(chunk_times),
                "tpot_s": tpot, "itl_p50_s": itl_p50, "itl_p99_s": itl_p99, "itl_max_s": itl_max,
                "error": err})
    if tag.get("workload") == "quality":
        row["text"] = "".join(text)
    return row


def _client(concurrency: int) -> httpx.AsyncClient:
    lim = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)
    return httpx.AsyncClient(limits=lim)


# A paid endpoint must never see an arrival rate computed from a bad measurement.
MAX_OPEN_LOOP_RPS = 100.0


async def closed_loop(ep: Endpoint, make_prompt: Callable[[int], str], concurrency: int, max_tokens: int,
                      duration_s: float, tag: dict) -> tuple[list[dict], float]:
    """`concurrency` workers for `duration_s`. Prompt i is make_prompt(i), so every arm gets the same prompts.
    Returns (rows, measured seconds); the measured window starts when the first wave has finished."""
    rows: list[dict] = []
    counter = iter(range(10**9))
    async with _client(concurrency) as client:
        t_start = time.perf_counter()
        deadline = t_start + duration_s
        warm_done: list[float] = []

        async def worker(w: int):
            first = True
            while time.perf_counter() < deadline:
                i = next(counter)
                r = await one_request(client, ep, make_prompt(i), max_tokens,
                                      {**tag, "concurrency": concurrency, "i": i, "warmup": first})
                if first:
                    warm_done.append(time.perf_counter())
                rows.append(r)
                first = False

        await asyncio.gather(*(worker(w) for w in range(concurrency)))
        t_end = time.perf_counter()
    return rows, measured_window(t_start, t_end, warm_done)


def measured_window(t_start: float, t_end: float, warm_done: list[float]) -> float:
    """Seconds from the end of the first wave to the end of the level.

    The first wave ends when 90% of workers have finished their first request. Waiting for every
    worker let one stalled first request shrink the window to almost nothing, which turned the
    request rate into a number in the millions (Together run, 2026-11-16: a few requests took 145 s
    while most took 1.5 s). Up to 9 workers this is the same as waiting for all of them."""
    if not warm_done:
        return max(t_end - t_start, 1e-9)
    done = sorted(warm_done)
    k = max(0, -(-len(done) * 9 // 10) - 1)   # index of the 90th percentile, rounded up
    return max(t_end - done[k], 1e-9)


async def open_loop(ep: Endpoint, make_prompt: Callable[[int], str], rate_rps: float, max_tokens: int,
                    duration_s: float, tag: dict, seed: int = 0) -> tuple[list[dict], float]:
    """Poisson arrivals at `rate_rps` for `duration_s`; waits for every request to finish."""
    if not 0 < rate_rps <= MAX_OPEN_LOOP_RPS:
        raise ValueError(f"open-loop rate {rate_rps:.1f} req/s is outside 0-{MAX_OPEN_LOOP_RPS}; "
                         "the closed-loop peak it came from is wrong")
    rng = random.Random(seed)
    tasks = []
    async with _client(512) as client:
        t0 = time.perf_counter()
        t_next = t0
        i = 0
        while t_next < t0 + duration_s:
            delay = t_next - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one_request(
                client, ep, make_prompt(i), max_tokens,
                {**tag, "rate_rps": rate_rps, "i": i, "warmup": False})))
            i += 1
            t_next += rng.expovariate(rate_rps)
        rows = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return list(rows), wall


async def _rtt(url: str, headers: dict, n: int) -> list[float]:
    out = []
    async with httpx.AsyncClient() as c:
        await c.get(url, headers=headers, timeout=30)   # opens the connection; not counted
        for _ in range(n):
            t = time.perf_counter()
            await c.get(url, headers=headers, timeout=30)
            out.append(time.perf_counter() - t)
    return out


def rtt(ep: Endpoint, n: int = 7) -> float:
    """Median round trip of a cheap GET on a warm connection: the network share of every TTFT."""
    # No health route on a hosted API: use a path that returns a small 404. GET /models is the wrong probe,
    # because Together's list is a 436 KB download (it read 232 ms and then 1,914 ms on the same day).
    url = ep.metrics_url.replace("/metrics", "/health") if ep.metrics_url else f"{ep.base_url}/rtt-probe"
    try:
        return statistics.median(asyncio.run(_rtt(url, ep.headers, n)))
    except httpx.HTTPError as e:   # a diagnostic only: a slow network must not end a paid phase
        print(f"  rtt: skipped ({type(e).__name__})")
        return float("nan")


_METRIC = re.compile(r'^(vllm:[a-z_]+?)(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$', re.M)
WANTED = {"vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
          "vllm:prefix_cache_queries", "vllm:prefix_cache_hits"}


def read_metrics(metrics_url: str | None) -> dict | None:
    """The vLLM counters this harness uses, summed over labels. None when there is no /metrics."""
    if not metrics_url:
        return None
    try:
        text = httpx.get(metrics_url, timeout=30).text
    except Exception:
        return None
    vals = {k: 0.0 for k in WANTED}
    for name, v in _METRIC.findall(text):
        if name in vals:
            vals[name] += float(v)
    return vals


def metrics_delta(before: dict | None, after: dict | None) -> dict:
    if not before or not after:
        return {}
    d = {k: after[k] - before[k] for k in WANTED}
    out = {}
    if d["vllm:time_to_first_token_seconds_count"] > 0:
        out["server_ttft_mean"] = d["vllm:time_to_first_token_seconds_sum"] / d["vllm:time_to_first_token_seconds_count"]
    if d["vllm:prefix_cache_queries"] > 0:
        out["prefix_cache_hit_rate"] = d["vllm:prefix_cache_hits"] / d["vllm:prefix_cache_queries"]
    return out
