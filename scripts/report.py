"""Turn a run's JSONL into results/<run_id>/summary.md, with sanity checks on the numbers.

    uv run python scripts/report.py final
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from providers.modal_vllm import ModalVLLM   # noqa: E402
from providers.together import Together       # noqa: E402


def read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def ms(x) -> str:
    return "n/a" if bad(x) else f"{x * 1000:,.0f}"


def num(x, fmt="{:,.0f}") -> str:
    return "n/a" if bad(x) else fmt.format(x)


def p95_cell(lv: dict, key: str, n_min: int) -> str:
    if bad(lv[key]):
        return f"n<{n_min}"
    if key == "ttft_p95":
        lo, hi = lv["ttft_p95_ci"]
        return f"{ms(lv[key])} [{ms(lo)}, {ms(hi)}]"
    return ms(lv[key])


def cost_cells(lv: dict, pricer: dict, util_cases: list[float]) -> str:
    p = pricer[lv["provider"]]
    if lv["provider"] == "together":
        c = p.usd_per_1k_requests(lv["prompt_tokens_mean"], lv["completion_tokens_mean"])
        return " / ".join([num(c, "${:.3f}")] * len(util_cases)) + " (posted price)"
    return " / ".join(num(p.usd_per_1k_requests(lv["req_s"], u), "${:.3f}") for u in util_cases)


def checks(levels: list[dict], rtts: dict) -> list[str]:
    """The patterns a sweep should show. A break gets a stated cause or a re-run."""
    notes = []
    by = {}
    for lv in levels:
        if lv["workload"] == "chat":
            by.setdefault((lv["provider"], lv["variant"]), []).append(lv)
    for (p, v), lvs in by.items():
        lvs.sort(key=lambda x: x["concurrency"])
        ttft = [x["ttft_p50"] for x in lvs]
        thr = [x["throughput_out_tok_s"] for x in lvs]
        if any(b < a * 0.9 for a, b in zip(ttft, ttft[1:])):
            notes.append(f"{p}/{v}: TTFT p50 fell as concurrency rose ({', '.join(ms(t) for t in ttft)} ms).")
        if any(b < a * 0.9 for a, b in zip(thr, thr[1:])):
            notes.append(f"{p}/{v}: throughput fell as concurrency rose ({', '.join(num(t) for t in thr)} tok/s).")
        errs = sum(x["errors"] for x in lvs)
        if errs:
            notes.append(f"{p}/{v}: {errs} failed requests in the chat sweep; samples: {lvs[-1].get('error_samples')}")
        short = [x for x in lvs if not bad(x["full_length_share"]) and x["full_length_share"] < 0.9]
        if short:
            notes.append(f"{p}/{v}: under 90% of responses reached max_tokens at concurrency "
                         f"{[x['concurrency'] for x in short]}; per-request TPS and cost use real lengths.")
        if any(x.get("usage_from_chunks") for x in lvs):
            notes.append(f"{p}/{v}: some responses had no usage block; token counts fell back to chunk counts.")
        for x in lvs:
            s = x.get("server_ttft_mean")
            r = rtts.get((p, v))
            if s and not bad(r) and not bad(x["ttft_mean"]):
                gap = x["ttft_mean"] - s
                if gap < 0 or gap > max(4 * r, 0.25):
                    notes.append(f"{p}/{v} c={x['concurrency']}: client TTFT mean {ms(x['ttft_mean'])} ms vs server "
                                 f"{ms(s)} ms; the gap ({ms(gap)} ms) is not explained by the {ms(r)} ms round trip.")
    for lv in levels:
        if lv["workload"] == "long_context" and "prefix_cache_hit_rate" in lv and lv["variant"] == "bf16" \
                and lv["prefix_cache_hit_rate"] < 0.5:
            notes.append(f"long_context bf16: prefix-cache hit rate only {lv['prefix_cache_hit_rate']:.0%}.")
    return notes or ["All chat sweeps follow the expected pattern: TTFT up, and throughput up or flat, as concurrency rises."]


def table(levels, key_cols, pricer, util_cases, n_min, slo) -> list[str]:
    head = ("| Provider | Variant | " + " | ".join(k for k, _ in key_cols) + " | n | Err | Prompt tok | Full len | "
            "TTFT p50 ms | TTFT p95 ms [95% CI] | Server TTFT mean ms | TPOT p50 ms | ITL p99 ms | "
            "Out tok/s | Out TPM | Req/s | Goodput | Prefix hit | $ per 1K req at "
            + " / ".join(f"{u:.0%}" for u in util_cases) + " busy |")
    out = [head, "|" + "---|" * (head.count("|") - 1)]
    for lv in levels:
        hit = lv.get("prefix_cache_hit_rate")
        cells = [lv["provider"], lv["variant"], *[str(lv.get(k2, "")) for _, k2 in key_cols],
                 str(lv["n"]), str(lv["errors"]), num(lv["prompt_tokens_mean"]), num(lv["full_length_share"], "{:.0%}"),
                 ms(lv["ttft_p50"]), p95_cell(lv, "ttft_p95", n_min), ms(lv.get("server_ttft_mean")),
                 num(lv["tpot_p50_ms"], "{:.1f}"), num(lv["itl_p99_ms_median"], "{:.1f}"),
                 num(lv["throughput_out_tok_s"]), num(lv["tpm_out"]), num(lv["req_s"], "{:.2f}"),
                 num(lv["goodput_share"], "{:.0%}"), "n/a" if hit is None else f"{hit:.0%}",
                 cost_cells(lv, pricer, util_cases)]
        out.append("| " + " | ".join(cells) + " |")
    return out


def main(run_id: str) -> None:
    d = ROOT / "results" / run_id
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    levels = read(d / "levels.jsonl")
    colds = read(d / "coldstarts.jsonl")
    rtt_rows = read(d / "rtt.jsonl")
    quality = read(d / "quality.jsonl")
    spend = [e for e in read(ROOT / "results" / "spend.jsonl") if e.get("run_id") == run_id]
    mcfg = cfg["providers"]["modal_vllm"]
    util = mcfg["utilization_cases"]
    n_min = cfg["min_n_for_p95"]
    slo = cfg["slo"]
    pricer = {"modal_vllm": ModalVLLM(mcfg, cfg["model"]["served_name"]),
              "together": Together(cfg["providers"]["together"], cfg["model"]["together_id"])}
    rtts = {(r["provider"], r["variant"]): r["rtt_s"] for r in rtt_rows}

    out = [f"# Run {run_id}", "",
           f"Self-hosted: vLLM on one Modal {mcfg['gpu']}, weights per variant in `config/poc.yaml`. "
           f"Together: `{cfg['model']['together_id']}` on its shared serverless fleet.",
           f"Goodput: share of measured requests with TTFT ≤ {slo['ttft_s']} s and end-to-end ≤ {slo['e2e_s']} s. "
           f"p95 is shown only when n ≥ {n_min}. Goodput uses client TTFT, so it includes the network path "
           f"from the test machine; compare server TTFT for GPU-side latency.", ""]
    if rtts:
        out += ["Network round trip from the test machine (median of 7, warm connection): "
                + "; ".join(f"{p}/{v} {ms(r)} ms" for (p, v), r in rtts.items())
                + ". Client TTFT includes this.", ""]

    sections = [("chat", "Chat: closed loop, about {i:,} tokens in, {o} out", [("Conc.", "concurrency")]),
                ("long_context", "Long context: one shared document, about {i:,} tokens in, {o} out", [("Conc.", "concurrency")]),
                ("open_loop", "Open loop: Poisson arrivals at a share of each arm's closed-loop peak", [("Load", "load_fraction"), ("Rate req/s", "rate_rps")])]
    for wl, title, keys in sections:
        rows = sorted([lv for lv in levels if lv["workload"] == wl],
                      key=lambda x: (x["provider"], x["variant"], x.get("concurrency") or x.get("load_fraction") or 0))
        if not rows:
            continue
        w = cfg["workloads"][wl]
        out += [f"## {title.format(i=w['input_tokens'], o=w['output_tokens'])}", ""]
        out += table(rows, keys, pricer, util, n_min, slo) + [""]

    if quality:
        out += [f"## Quality check: {cfg['workloads']['quality']['n']} arithmetic questions, temperature 0", "", "| Provider | Variant | Exact |", "|---|---|---|"]
        arms = {}
        for q in quality:
            arms.setdefault((q["provider"], q["variant"]), []).append(q)
        for (p, v), qs in arms.items():
            out.append(f"| {p} | {v} | {sum(q['correct'] for q in qs)}/{len(qs)} |")
        out.append("")
        base = {q["i"]: (q.get("text") or "").strip() for q in arms.get(("modal_vllm", "bf16"), [])}
        for (p, v), qs in arms.items():
            if base and (p, v) != ("modal_vllm", "bf16"):
                same = sum(base.get(q["i"]) == (q.get("text") or "").strip() for q in qs)
                out.append(f"- {p}/{v} gives the same answer as bf16 on {same}/{len(qs)} questions.")
        out += ["", "Two-digit multiplication answered with the number only is hard for a 7B model, so exact "
                "scores are low on every arm. The test cannot separate small quality changes at n=50; "
                "agreement with bf16 is the more direct check on quantization.", ""]

    if colds:
        out += ["## Cold starts (one observation each)", "",
                "| Label | GPU | vLLM | Total to first token s | Schedule + boot s | Weights + engine s | First request TTFT ms |",
                "|---|---|---|---|---|---|---|"]
        for c in colds:
            out.append(f"| {c['label']} | {c.get('gpu', 'n/a')} | {c.get('vllm', 'n/a')} | "
                       f"{num(c['total_to_first_token_s'], '{:.1f}')} | {num(c.get('schedule_and_boot_s'), '{:.1f}')} | "
                       f"{num(c.get('weights_and_engine_s'), '{:.1f}')} | {ms(c['first_request_ttft_s'])} |")
        out += ["", "The boot split subtracts this machine's clock from the container's clock, so it carries any "
                "clock difference between them. The second bf16 start has a warm image cache and a warm compile "
                "cache, so it shows both effects together.", ""]

    out += ["## Sanity checks", ""] + [f"- {n}" for n in checks(levels, rtts)] + [""]
    out += ["## Spend for this run", "", "| Phase | USD | Basis |", "|---|---|---|"]
    out += [f"| {s['phase']} | {s['usd']:.3f} | {s['note']} |" for s in spend]
    out += [f"| **Total** | **{sum(s['usd'] for s in spend):.2f}** | |", ""]
    out += ["How to read cost: self-hosted cost per 1,000 requests is the container price divided by the measured "
            "request rate, at the share of time the GPU is busy. Together's is its posted input and output price "
            "at the measured token counts, and does not depend on load. Together's throughput is what this client "
            "saw on a shared fleet with rate limits, not the capacity of one GPU.", ""]
    (d / "summary.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
