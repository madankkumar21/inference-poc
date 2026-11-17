"""Offline tests: the whole measurement path against a fake server with known timings. No network, no spend."""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml

from bench import prompts as P
from bench.cost import Ledger, OverBudget
from bench.load import closed_loop, metrics_delta, one_request, open_loop, read_metrics
from bench.stats import summarize
from providers.base import Endpoint
from providers.modal_vllm import ModalVLLM
from providers.together import Together
from tests import fake_server

SLO = {"ttft_s": 1.0, "e2e_s": 15.0}


@pytest.fixture()
def server():
    st = fake_server.State(ttft=0.05, tpot=0.002, queue_per_inflight=0.01)
    srv, url = fake_server.start(st)
    yield st, url
    srv.shutdown()


def ep_for(url, self_hosted=True, key="test-key"):
    return Endpoint(provider="modal_vllm" if self_hosted else "together", variant="bf16" if self_hosted else "serverless",
                    base_url=f"{url}/v1", model="fake/model", headers={"Authorization": f"Bearer {key}"},
                    metrics_url=f"{url}/metrics" if self_hosted else None, honours_min_tokens=self_hosted)


def test_one_request_timings_and_fields(server):
    st, url = server
    import httpx

    async def go():
        async with httpx.AsyncClient() as c:
            return await one_request(c, ep_for(url), "hello " * 50, 40, {"workload": "chat"})
    r = asyncio.run(go())
    assert r["error"] is None
    assert r["completion_tokens"] == 40 and r["usage_source"] == "usage"
    assert r["finish_reason"] == "length"
    assert 0.04 < r["ttft_s"] < 0.5
    assert r["tpot_s"] is not None and 0.001 < r["tpot_s"] < 0.05
    assert r["itl_p99_s"] >= r["itl_p50_s"]
    assert st.bodies[-1]["min_tokens"] == 40          # vLLM arm forces full length
    assert st.bodies[-1]["stream_options"] == {"include_usage": True}


def test_bad_key_is_an_error_not_a_crash(server):
    _, url = server
    import httpx

    async def go():
        async with httpx.AsyncClient() as c:
            return await one_request(c, ep_for(url, key="wrong"), "hi", 5, {"workload": "chat"})
    r = asyncio.run(go())
    assert r["error"].startswith("http 401")


def test_missing_usage_falls_back_to_chunks():
    st = fake_server.State(usage=False)
    srv, url = fake_server.start(st)
    try:
        import httpx

        async def go():
            async with httpx.AsyncClient() as c:
                return await one_request(c, ep_for(url, self_hosted=False), "hi there", 12, {"workload": "chat"})
        r = asyncio.run(go())
        assert r["prompt_tokens"] is None and r["completion_tokens"] == 12 and r["usage_source"] == "chunk_count"
        assert "min_tokens" not in st.bodies[-1]       # serverless arm does not send vLLM-only fields
    finally:
        srv.shutdown()


def test_closed_loop_drops_first_wave_and_queueing_shows(server):
    _, url = server
    ep = ep_for(url)
    mk = lambda i: P.chat_prompt(200, 1.3, P.seed_for("chat", 1, i))  # noqa: E731
    r1, w1 = asyncio.run(closed_loop(ep, mk, 1, 20, 1.0, {"workload": "chat"}))
    r8, w8 = asyncio.run(closed_loop(ep, mk, 8, 20, 1.0, {"workload": "chat"}))
    assert sum(r["warmup"] for r in r1) == 1 and sum(r["warmup"] for r in r8) == 8
    s1, s8 = summarize(r1, w1, SLO, 100), summarize(r8, w8, SLO, 100)
    assert s8["ttft_p50"] > s1["ttft_p50"]                      # the fake queues, so TTFT must rise
    assert s8["throughput_out_tok_s"] > s1["throughput_out_tok_s"]
    assert s1["errors"] == 0 and s1["goodput_share"] == 1.0
    assert s1["n"] < 100 and s1["ttft_p95"] != s1["ttft_p95"]  # NaN: p95 hidden below min n


def test_same_prompts_for_every_arm():
    a = [P.chat_prompt(1000, 1.3, P.seed_for("chat", 16, i)) for i in range(5)]
    b = [P.chat_prompt(1000, 1.3, P.seed_for("chat", 16, i)) for i in range(5)]
    assert a == b and len(set(a)) == 5
    assert a[0][:60] != a[1][:60]                               # unique prefixes: no cache hits in chat


def test_open_loop_rate(server):
    _, url = server
    mk = lambda i: P.chat_prompt(100, 1.3, P.seed_for("open", 0.5, i))  # noqa: E731
    rows, wall = asyncio.run(open_loop(ep_for(url), mk, 20.0, 10, 2.0, {"workload": "open_loop"}, seed=1))
    assert 25 <= len(rows) <= 60                                # Poisson at 20/s for 2 s


def test_metrics_and_prefix_cache(server):
    _, url = server
    ep = ep_for(url)
    doc = P.SharedDocument(1500, 1.3)
    import httpx

    async def prime():
        async with httpx.AsyncClient() as c:
            return await one_request(c, ep, doc.prompt(-1), 4, {"workload": "long_context_prime"})
    asyncio.run(prime())
    before = read_metrics(ep.metrics_url)
    rows, w = asyncio.run(closed_loop(ep, doc.prompt, 4, 10, 1.0, {"workload": "long_context"}))
    after = read_metrics(ep.metrics_url)
    d = metrics_delta(before, after)
    assert d["prefix_cache_hit_rate"] == 1.0                    # primed, so every measured request hits
    s = summarize(rows, w, SLO, 100)
    assert abs(d["server_ttft_mean"] - s["ttft_mean"]) < 0.05   # second method agrees with the client


def test_ledger_is_repo_wide_and_stops(tmp_path):
    path = tmp_path / "spend.jsonl"
    a = Ledger(1.0, path, "run-a")
    a.add("together", 0.7)
    b = Ledger(1.0, path, "run-b")                              # a new run id sees the old spend
    assert b.spent == pytest.approx(0.7)
    with pytest.raises(OverBudget):
        b.check("bf16", 0.5)
    with pytest.raises(OverBudget):
        b.check_running("bf16", 0.31)


def test_costs_per_1k_requests():
    cfg = yaml.safe_load(open("config/poc.yaml"))
    m = ModalVLLM(cfg["providers"]["modal_vllm"], "x")
    t = Together(cfg["providers"]["together"], "y")
    per_s = m.usd_per_sec()
    assert per_s == pytest.approx(0.000542 + 2 * 0.00003942 + 24 * 0.00000667)
    assert m.usd_per_1k_requests(2.0, 1.0) == pytest.approx(per_s / 2 * 1000)
    assert m.usd_per_1k_requests(2.0, 0.3) == pytest.approx(per_s / 0.6 * 1000)
    tc = cfg["providers"]["together"]   # input and output are priced separately
    assert t.usd_per_1k_requests(1000, 250) == pytest.approx(
        1000 * (1000 * tc["usd_per_m_input"] + 250 * tc["usd_per_m_output"]) / 1e6)


def test_quality_items_have_known_answers(server):
    _, url = server
    import httpx
    items = P.quality_items(5)

    async def go():
        async with httpx.AsyncClient() as c:
            return [await one_request(c, ep_for(url), q, 16, {"workload": "quality"}) for q, _ in items]
    rows = asyncio.run(go())
    assert [r["text"] for r in rows] == [a for _, a in items]
    assert P.quality_items(5) == items                          # fixed seed


def test_end_to_end_runner_and_report(server, tmp_path, monkeypatch):
    """Run the runner's measurement steps on the fake server, then build the report."""
    _, url = server
    from scripts import report, run_poc

    monkeypatch.setattr(run_poc, "ROOT", tmp_path)
    monkeypatch.setattr(run_poc, "CALIBRATION", tmp_path / "results" / "calibration.json")
    monkeypatch.setattr(report, "ROOT", tmp_path)
    cfg = yaml.safe_load(open("config/poc.yaml"))
    cfg["workloads"]["chat"].update({"concurrency": [1, 4], "duration_s": 1})
    cfg["workloads"]["long_context"].update({"input_tokens": 1500, "duration_s": 1})
    cfg["workloads"]["open_loop"].update({"duration_s": 1})
    cfg["workloads"]["quality"]["n"] = 5
    run = run_poc.Run("t", cfg)
    ep = ep_for(url)
    tpw = run.tokens_per_word(ep)
    assert 1.0 < tpw < 1.6
    run.record_rtt(ep)
    run.quality(ep)
    guard = lambda rows: None  # noqa: E731
    run.chat_sweep(ep, tpw, guard)
    run.long_context(ep, tpw, guard)
    run.open_loop_runs(ep, tpw, guard)
    report.main("t")
    md = (tmp_path / "results" / "t" / "summary.md").read_text()
    for heading in ("## Chat", "## Long context", "## Open loop", "## Quality check", "## Sanity checks"):
        assert heading in md
    assert "5/5" in md
    levels = [json.loads(x) for x in (tmp_path / "results" / "t" / "levels.jsonl").read_text().splitlines()]
    assert {lv["workload"] for lv in levels} == {"chat", "long_context", "open_loop"}



def test_measured_window_ignores_a_stalled_first_request():
    from bench.load import measured_window
    # 64 workers: 63 finish warm-up at t=2 s, one stalls until the level ends at t=60 s.
    assert measured_window(0.0, 60.0, [2.0] * 63 + [60.0]) == pytest.approx(58.0)
    # Few workers: same as waiting for all of them.
    assert measured_window(0.0, 60.0, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(56.0)
    assert measured_window(0.0, 60.0, []) == pytest.approx(60.0)


def test_open_loop_refuses_an_absurd_rate():
    import asyncio
    from bench.load import open_loop
    from providers.base import Endpoint
    ep = Endpoint("x", "y", "http://127.0.0.1:1/v1", "m", {})
    with pytest.raises(ValueError):
        asyncio.run(open_loop(ep, lambda i: "p", 5_500_000.0, 8, 1.0, {}))


def test_a_request_past_the_deadline_is_an_error(monkeypatch):
    import httpx
    from bench import load
    st = fake_server.State(ttft=0.6, tpot=0.002, queue_per_inflight=0.0)
    srv, url = fake_server.start(st)
    monkeypatch.setattr(load, "REQUEST_DEADLINE_S", 0.2)

    async def go():
        async with httpx.AsyncClient() as c:
            return await load.one_request(c, ep_for(url), "hello", 8, {"workload": "chat"})
    try:
        r = asyncio.run(go())
    finally:
        srv.shutdown()
    assert r["error"].startswith("deadline")
    assert r["e2e_s"] < 0.5
