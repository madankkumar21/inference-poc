"""Percentiles, a bootstrap interval on p95, goodput, and per-level summaries."""

from __future__ import annotations

import math

import numpy as np


def _clean(xs) -> list[float]:
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def pct(xs, q: float) -> float:
    xs = _clean(xs)
    return float(np.percentile(xs, q)) if xs else float("nan")


def boot_ci(xs, q: float = 95, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    """Percentile-bootstrap interval for the q-th percentile. Only meaningful with enough points."""
    xs = np.asarray(_clean(xs), dtype=float)
    if len(xs) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(xs), size=(n_boot, len(xs)))
    boots = np.percentile(xs[idx], q, axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize(rows: list[dict], window_s: float, slo: dict, min_n_for_p95: int = 100) -> dict:
    """Summary of the measured rows (warm-up rows are left out). Throughput counts only requests
    that finished inside the measured window, so a long tail does not inflate it."""
    measured = [r for r in rows if not r.get("warmup")]
    ok = [r for r in measured if not r.get("error")]
    ttft = [r["ttft_s"] for r in ok]
    e2e = [r["e2e_s"] for r in ok]
    tpot = [r["tpot_s"] for r in ok]
    out_tok = sum(r.get("completion_tokens") or 0 for r in ok)
    in_tok = [r.get("prompt_tokens") for r in ok]
    met = [r for r in ok if r["ttft_s"] <= slo["ttft_s"] and r["e2e_s"] <= slo["e2e_s"]]
    full = [r for r in ok if r.get("completion_tokens") and r["completion_tokens"] >= r["max_tokens"]]
    enough = len(ok) >= min_n_for_p95
    p95_lo, p95_hi = boot_ci(ttft, 95) if enough else (float("nan"), float("nan"))
    req_s = len(ok) / window_s if window_s > 0 else float("nan")
    return {
        "n": len(measured),
        "warmup_dropped": len(rows) - len(measured),
        "errors": len(measured) - len(ok),
        "error_samples": sorted({r["error"][:120] for r in measured if r.get("error")})[:3],
        "prompt_tokens_mean": float(np.mean(_clean(in_tok))) if _clean(in_tok) else float("nan"),
        "completion_tokens_mean": out_tok / len(ok) if ok else float("nan"),
        "full_length_share": len(full) / len(ok) if ok else float("nan"),
        "usage_from_chunks": sum(1 for r in ok if r.get("usage_source") == "chunk_count"),
        "ttft_p50": pct(ttft, 50),
        "ttft_p95": pct(ttft, 95) if enough else float("nan"),
        "ttft_p95_ci": [p95_lo, p95_hi],
        "ttft_mean": float(np.mean(ttft)) if ttft else float("nan"),
        "e2e_p50": pct(e2e, 50),
        "e2e_p95": pct(e2e, 95) if enough else float("nan"),
        "tpot_p50_ms": pct(tpot, 50) * 1000,
        "itl_p99_ms_median": pct([r.get("itl_p99_s") for r in ok], 50) * 1000,
        "per_request_tps_p50": (1 / pct(tpot, 50)) if _clean(tpot) else float("nan"),
        "throughput_out_tok_s": out_tok / window_s if window_s > 0 else float("nan"),
        "tpm_out": out_tok / window_s * 60 if window_s > 0 else float("nan"),
        "req_s": req_s,
        "goodput_share": len(met) / len(measured) if measured else float("nan"),
        "goodput_req_s": len(met) / window_s if window_s > 0 else float("nan"),
        "window_s": window_s,
    }
