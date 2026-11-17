"""Run the POC: for each phase, start the endpoint, warm it, measure, and stop it.

    uv run python scripts/run_poc.py --phases smoke
    uv run python scripts/run_poc.py --phases together,bf16,fp8,nocache --run-id final

Phases:
  smoke     bf16 on Modal: one request, server version and GPU, then stop. Run this first.
  together  Together serverless: model check, quality, chat sweep, long context, open loop
  bf16      vLLM defaults on Modal: cold start x2, quality, chat sweep, long context (prefix cache on), open loop
  fp8       pre-quantized FP8 checkpoint: cold start, quality, chat sweep, open loop
  nocache   bf16 with prefix caching off: long context only (the other half of the prefix-cache test)

Every Modal phase stops its app in a finally block and then checks Modal no longer lists it.
Spend goes to one ledger for the repo (results/spend.jsonl), checked before each phase and after each level.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench import prompts as P                                                    # noqa: E402
from bench.cost import Ledger, OverBudget                                         # noqa: E402
from bench.load import closed_loop, metrics_delta, one_request, open_loop, read_metrics, rtt  # noqa: E402
from bench.stats import summarize                                                 # noqa: E402
from providers.base import Endpoint                                               # noqa: E402

# Rough spend per phase, used only for the check before a phase starts.
EXPECTED_PHASE_USD = {"smoke": 0.4, "together": 3.0, "bf16": 1.5, "fp8": 0.8, "nocache": 0.4}
CALIBRATION = ROOT / "results" / "calibration.json"


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "config" / "poc.yaml").read_text())


def jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


class Run:
    def __init__(self, run_id: str, cfg: dict):
        self.dir = ROOT / "results" / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.slo = cfg["slo"]
        self.ledger = Ledger(cfg["budget_usd"], ROOT / "results" / "spend.jsonl", run_id)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.peak_req_s: dict[str, float] = {}

    # ---------- shared steps ----------
    async def _one(self, ep, prompt, max_tokens, tag):
        async with httpx.AsyncClient() as c:
            return await one_request(c, ep, prompt, max_tokens, tag)

    def tokens_per_word(self, ep: Endpoint) -> float:
        """Measured once and reused by every arm, so all arms get identical prompts."""
        if CALIBRATION.exists():
            return json.loads(CALIBRATION.read_text())["tokens_per_word"]
        prompt = P.chat_prompt(540, P.DEFAULT_TOKENS_PER_WORD, seed=999)
        row = asyncio.run(self._one(ep, prompt, 16, {"workload": "calibrate"}))
        if row["error"] or not row["prompt_tokens"]:
            raise RuntimeError(f"calibration needs usage.prompt_tokens; got error={row['error']} usage={row['prompt_tokens']}")
        tpw = P.calibrate(len(prompt.split()), row["prompt_tokens"])
        CALIBRATION.write_text(json.dumps({"tokens_per_word": tpw, "from": ep.provider,
                                           "words": len(prompt.split()), "prompt_tokens": row["prompt_tokens"]}))
        print(f"  calibrated on {ep.provider}: {tpw:.3f} tokens/word")
        return tpw

    def record_rtt(self, ep: Endpoint) -> None:
        r = rtt(ep)
        jsonl(self.dir / "rtt.jsonl", {"provider": ep.provider, "variant": ep.variant, "rtt_s": r})
        print(f"  network round trip {r * 1000:.0f} ms (median of 7, warm connection)")

    def save_level(self, ep: Endpoint, workload: str, rows, window, extra: dict, before, after) -> dict:
        for r in rows:
            jsonl(self.dir / "rows.jsonl", r)
        s = summarize(rows, window, self.slo, self.cfg["min_n_for_p95"])
        s.update({"provider": ep.provider, "variant": ep.variant, "workload": workload, **extra,
                  **metrics_delta(before, after)})
        jsonl(self.dir / "levels.jsonl", s)
        hit = f" | prefix hit {s['prefix_cache_hit_rate']:.0%}" if "prefix_cache_hit_rate" in s else ""
        srv = f" | server TTFT mean {s['server_ttft_mean'] * 1000:.0f} ms" if "server_ttft_mean" in s else ""
        print(f"  {ep.provider}/{ep.variant} {workload} {extra}: n={s['n']} err={s['errors']} "
              f"TTFT p50 {s['ttft_p50'] * 1000:.0f} ms | TPOT p50 {s['tpot_p50_ms']:.1f} ms | "
              f"{s['throughput_out_tok_s']:.0f} out tok/s | {s['req_s']:.2f} req/s | goodput {s['goodput_share']:.0%}"
              f"{srv}{hit}")
        if s["errors"]:
            print(f"    errors: {s['error_samples']}")
        return s

    def chat_sweep(self, ep: Endpoint, tpw: float, guard) -> None:
        w = self.cfg["workloads"]["chat"]
        levels = w["concurrency"]
        if not ep.honours_min_tokens:   # serverless: stop where we would measure the rate limit
            levels = [c for c in levels if c <= w["together_max_concurrency"]]
        peak = 0.0
        for c in levels:
            mk = lambda i, c=c: P.chat_prompt(w["input_tokens"], tpw, P.seed_for("chat", c, i))  # noqa: E731
            before = read_metrics(ep.metrics_url)
            rows, window = asyncio.run(closed_loop(ep, mk, c, w["output_tokens"], w["duration_s"], {"workload": "chat"}))
            after = read_metrics(ep.metrics_url)
            s = self.save_level(ep, "chat", rows, window, {"concurrency": c}, before, after)
            if window < 0.5 * w["duration_s"]:   # most of the level was warm-up: the rate is not usable
                print(f"    level c={c} ignored for the peak: measured window {window:.1f} s")
            else:
                peak = max(peak, s["req_s"])
            guard(rows)
        self.peak_req_s[ep.variant] = peak

    def long_context(self, ep: Endpoint, tpw: float, guard) -> None:
        w = self.cfg["workloads"]["long_context"]
        doc = P.SharedDocument(w["input_tokens"], tpw)
        # Priming request: puts the shared document in the prefix cache before the measured window.
        prime = asyncio.run(self._one(ep, doc.prompt(-1), 8, {"workload": "long_context_prime"}))
        jsonl(self.dir / "rows.jsonl", prime)
        for c in w["concurrency"]:
            before = read_metrics(ep.metrics_url)
            rows, window = asyncio.run(closed_loop(ep, doc.prompt, c, w["output_tokens"], w["duration_s"],
                                                   {"workload": "long_context"}))
            after = read_metrics(ep.metrics_url)
            self.save_level(ep, "long_context", rows, window, {"concurrency": c}, before, after)
            guard(rows)

    def open_loop_runs(self, ep: Endpoint, tpw: float, guard) -> None:
        w = self.cfg["workloads"]["open_loop"]
        peak = self.peak_req_s.get(ep.variant)
        if not peak:
            print("  open loop skipped: no closed-loop peak for this arm")
            return
        for f in w["load_fractions"]:
            rate = peak * f
            mk = lambda i, f=f: P.chat_prompt(w["input_tokens"], tpw, P.seed_for("open", f, i))  # noqa: E731
            before = read_metrics(ep.metrics_url)
            rows, wall = asyncio.run(open_loop(ep, mk, rate, w["output_tokens"], w["duration_s"],
                                               {"workload": "open_loop"}, seed=int(f * 100)))
            after = read_metrics(ep.metrics_url)
            self.save_level(ep, "open_loop", rows, wall, {"load_fraction": f, "rate_rps": round(rate, 3)},
                            before, after)
            guard(rows)

    def quality(self, ep: Endpoint) -> None:
        items = P.quality_items(self.cfg["workloads"]["quality"]["n"])

        async def go():
            async with httpx.AsyncClient() as c:
                sem = asyncio.Semaphore(8)

                async def q(i, prompt):
                    async with sem:
                        return await one_request(c, ep, prompt, 16, {"workload": "quality", "i": i})
                return await asyncio.gather(*(q(i, p) for i, (p, _) in enumerate(items)))
        rows = asyncio.run(go())
        right = 0
        for r, (_, ans) in zip(rows, items):
            r["expected"] = ans
            r["correct"] = (r.get("text") or "").strip().rstrip(".") == ans
            right += r["correct"]
            jsonl(self.dir / "quality.jsonl", r)
        print(f"  quality: {right}/{len(items)} arithmetic answers exact")

    # ---------- Modal ----------
    def cold_start(self, ep: Endpoint, label: str) -> dict:
        """Time a request against a scaled-to-zero app: wait for /health, then time the first token."""
        base = ep.base_url.removesuffix("/v1")
        t_send = time.time()
        deadline = t_send + 25 * 60
        while True:
            try:
                r = httpx.get(f"{base}/health", timeout=30, follow_redirects=True)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise RuntimeError("endpoint not healthy after 25 minutes")
            time.sleep(2)
        t_health = time.time()
        row = asyncio.run(self._one(ep, P.chat_prompt(1000, P.DEFAULT_TOKENS_PER_WORD, 4242), 32,
                                    {"workload": f"coldstart:{label}"}))
        rec = {"provider": ep.provider, "variant": ep.variant, "label": label, "t_send": t_send,
               "client_health_s": t_health - t_send, "first_request_ttft_s": row["ttft_s"],
               "total_to_first_token_s": (t_health - t_send) + (row["ttft_s"] or float("nan")),
               "error": row["error"]}
        try:
            import modal
            d = modal.Dict.from_name("inference-poc-timings")
            mine = [v for _, v in d.items() if v["variant"] == ep.variant and v["container_start"] >= t_send - 5]
            if mine:
                m = min(mine, key=lambda v: v["container_start"])
                rec.update({"gpu": m["gpu"], "vllm": m["vllm"], "weights": m.get("weights"),
                            "schedule_and_boot_s": m["container_start"] - t_send,
                            "weights_and_engine_s": m["server_ready"] - m["container_start"]})
        except Exception as e:
            rec["timings_error"] = repr(e)
        jsonl(self.dir / "coldstarts.jsonl", rec)
        print(f"  cold start {label}: {rec['total_to_first_token_s']:.1f} s to first token "
              f"(boot {rec.get('schedule_and_boot_s', float('nan')):.1f} s, "
              f"weights+engine {rec.get('weights_and_engine_s', float('nan')):.1f} s, "
              f"GPU {rec.get('gpu', '?')}, vLLM {rec.get('vllm', '?')})")
        return rec

    def modal_phase(self, prov, phase: str, variant: str, steps: list[str]) -> None:
        self.ledger.check(phase, EXPECTED_PHASE_USD[phase])
        state = {"t_first": None}

        def billed() -> float:
            return prov.run_cost_usd([], time.time() - state["t_first"]) if state["t_first"] else 0.0

        def guard(_rows):
            self.ledger.check_running(phase, billed())

        try:
            ep = prov.start(variant)
            state["t_first"] = time.time()
            self.cold_start(ep, f"{variant}-1")
            if phase == "smoke":
                r = asyncio.run(self._one(ep, "Say hello in five words.", 16, {"workload": "smoke"}))
                print(f"  smoke: error={r['error']} ttft={r['ttft_s']} tokens={r['completion_tokens']} "
                      f"metrics={'ok' if read_metrics(ep.metrics_url) else 'missing'}")
                return
            self.record_rtt(ep)
            tpw = self.tokens_per_word(ep)
            if "quality" in steps:
                self.quality(ep)
            if "chat" in steps:
                self.chat_sweep(ep, tpw, guard)
            if "long_context" in steps:
                self.long_context(ep, tpw, guard)
            if "open_loop" in steps:
                self.open_loop_runs(ep, tpw, guard)
            if "coldstart2" in steps:
                prov.stop(variant)
                self.ledger.add(phase, billed(), "first container, container-seconds upper bound")
                state["t_first"] = None
                time.sleep(10)
                ep = prov.start(variant)
                state["t_first"] = time.time()
                self.cold_start(ep, f"{variant}-2-warm-caches")
        finally:
            prov.stop(variant)
            if state["t_first"] is not None:
                self.ledger.add(phase, billed(), "container-seconds, upper bound")

    def together_phase(self, prov) -> None:
        self.ledger.check("together", EXPECTED_PHASE_USD["together"])
        info = prov.check_model()
        jsonl(self.dir / "together_model.jsonl", info)
        print(f"  Together lists {prov.model_id}: {json.dumps({k: info.get(k) for k in ('type', 'context_length', 'pricing')})}")
        ep = prov.start()
        all_rows: list[dict] = []

        def guard(rows):
            all_rows.extend(rows)
            self.ledger.check_running("together", prov.run_cost_usd(all_rows))

        try:
            self.record_rtt(ep)
            tpw = self.tokens_per_word(ep)
            self.quality(ep)
            self.chat_sweep(ep, tpw, guard)
            self.long_context(ep, tpw, guard)
            self.open_loop_runs(ep, tpw, guard)
        finally:
            self.ledger.add("together", prov.run_cost_usd(all_rows), "tokens at posted price (sweeps only)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="together,bf16,fp8,nocache")
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y-%m-%d_%H%M"))
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_cfg()
    run = Run(args.run_id, cfg)
    print(f"run {args.run_id} -> {run.dir}; repo spend so far ${run.ledger.spent:.2f} of ${cfg['budget_usd']:.2f}")

    from providers.modal_vllm import ModalVLLM
    from providers.together import Together
    modal_p = ModalVLLM(cfg["providers"]["modal_vllm"], cfg["model"]["served_name"])
    together_p = Together(cfg["providers"]["together"], cfg["model"]["together_id"])

    full = ["quality", "chat", "long_context", "open_loop"]
    plan = {
        "smoke": lambda: run.modal_phase(modal_p, "smoke", "bf16", []),
        "together": lambda: run.together_phase(together_p),
        "bf16": lambda: run.modal_phase(modal_p, "bf16", "bf16", full + ["coldstart2"]),
        "fp8": lambda: run.modal_phase(modal_p, "fp8", "fp8", ["quality", "chat", "open_loop"]),
        "nocache": lambda: run.modal_phase(modal_p, "nocache", "bf16-nocache", ["long_context"]),
    }
    for ph in args.phases.split(","):
        print(f"== phase {ph}")
        try:
            plan[ph]()
        except OverBudget as e:
            print(f"STOP: {e}")
            break
    print(f"done. repo spend ${run.ledger.spent:.2f}. report: uv run python scripts/report.py {args.run_id}")


if __name__ == "__main__":
    main()
