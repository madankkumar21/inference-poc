# inference-poc

A 48-hour inference proof of concept (POC), run the way a solutions architect runs one for a customer. It serves an open-weight model on self-hosted vLLM, sweeps load, applies one tuning change at a time, and prices the result per 1,000 requests. The benchmark code only calls an OpenAI-compatible endpoint, so any provider fits. I ran it on [Modal](https://modal.com) with one NVIDIA L40S. A Together serverless arm is built and tested offline, but it has not run yet.

**Why this exists.** I have benchmarked search and AI pipelines for customers with large media libraries. I had not stood up a GPU serving stack myself. So I took the standard POC loop: traffic shape, endpoint, benchmark, tune, price. I ran it end to end on real hardware and measured each step twice where I could.

## What I tested

Model: Qwen2.5-7B-Instruct. Server: vLLM 0.11.0 on one L40S (48 GB). Chat traffic: about 1,000 tokens in and 250 out per request. Full tables with every column: [results/final/summary.md](results/final/summary.md).

| # | What a customer asks | What I measured | Evidence |
|---|---|---|---|
| 1 | How much traffic does one GPU take? | bf16: 508 output tok/s at 16 concurrent requests, 1,803 at 128. No errors at any level, and every response reached its full 250 tokens. | Closed-loop sweep at 1, 4, 16, 32, 64 and 128; 60 s per level; first wave dropped as warm-up |
| 2 | Does FP8 help? | **1.5–1.7x throughput at concurrency 1–32**, 1.4x at 64, 1.15x at 128. Time per output token at concurrency 1: 21.1 ms (bf16), 14.2 ms (fp8). | Same sweep and prompts with a pre-quantized FP8 checkpoint (`RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic`) |
| 3 | Does FP8 change the answers? | fp8 gave the same answer as bf16 on 38 of 50 fixed arithmetic questions. The differences were near misses (median error 0.2%). Exact scores: bf16 21/50, fp8 19/50. | Temperature 0, same 50 questions on each arm. Weak evidence: see "What I did not test" |
| 4 | What does prefix caching buy on a long shared document? | **Server time to first token (TTFT) 44 ms with the cache, 1,238 ms without: 28x.** Cache hit rate 99.5%. Throughput 151 against 116 tok/s. | 8,000-token shared document at 4 concurrent requests, one priming request first; hit rate from vLLM's own counters |
| 5 | What happens under real arrival patterns? | fp8 at half of its peak rate (4.2 req/s): TTFT p50 289 ms, p95 602 ms. At 80% of peak, p95 rises to 1,312 ms. | Open loop: random (Poisson) arrivals at 0.5 and 0.8 of each arm's closed-loop peak |
| 6 | How long is a cold start? | 68 s from request to first token on fresh caches. 51 s with warm image and compile caches. 95 s for the first FP8 start. | Scale from zero; container clock vs client clock gives the split |
| 7 | What does it cost? | fp8 at 16 concurrent requests: **$0.24 per 1,000 requests at 100% busy, $0.81 at 30% busy.** bf16 at the same load: $0.39 and $1.28. | Modal's per-second L40S, CPU and memory price, divided by measured requests per second |

**Two measurements of TTFT, and why the second one matters.** The client measured TTFT from this machine, and vLLM measured it on the server through `/metrics`. On the bf16 container the two disagreed by up to 920 ms: 995 ms at the client against 73 ms at the server. The network round trip to that container was 327 ms, against 59 ms for the fp8 container. So most client TTFT on that container was network and proxy time, not GPU time. For GPU-side comparisons I use server TTFT. I did not isolate the cause of the gap.

What I took from it, as advice I would give a customer:
- **Measure latency on the server as well as the client.** Without the server's histogram, this POC would have reported a 1-second TTFT for a model that answers in 73 ms. Run the load client in the same region as the endpoint, or report both numbers.
- **Try FP8 first on Ada or Hopper GPUs.** It was the largest single gain here: 1.5–1.7x throughput at the concurrency levels where latency is still usable. Check answer agreement on the customer's own prompts before switching, not on a public benchmark.
- **For a long shared context, make sure prefix caching is on and the shared part comes first.** RAG over one document, long system prompts and multi-turn chat all fit this pattern. One request whose prefix differs at the start gets none of the benefit.
- **Price at the real busy share, not at 100%.** One L40S running fp8 at 16 concurrent requests costs $0.24 per 1,000 requests when kept busy. Together's posted price for this model works out to $0.38 per 1,000 requests at the same token counts. So self-hosting wins only above about 63% busy. Spiky or low traffic favors serverless, and steady high traffic favors a dedicated GPU.
- **Pick the operating point from the latency target, then read the cost.** At a 1 s TTFT target, fp8 kept 99% of requests inside it at 16 concurrent requests and 43% at 32. The cheapest row in the table is at 128, where no request met the target.
- **Plan for a minute of cold start.** Keep one warm container for interactive traffic, or accept about 50–95 s on the first request after scale to zero.

## What I did not test

- **Together, measured.** The Together arm is built and passes the offline tests. It needs a funded API key, so it has not run. The Together price above is the posted price, not a measured one. Together also serves on different GPUs than one L40S, so a later comparison will not be matched on hardware.
- **Answer quality, properly.** 50 arithmetic questions with "number only" answers is too hard for a 7B model and too small to detect a small change. A real POC uses the customer's prompts and a task-specific grader.
- **The bf16 vs fp8 goodput gap.** Goodput (the share of requests that meet the latency target) uses client TTFT. The bf16 container had a much slower network path, so its goodput is lower for reasons that have nothing to do with FP8. The bf16 container also showed 290 ms gaps between some streamed tokens, against about 30 ms on fp8 at concurrency 1 and 4. This is probably the same network path, but I did not prove it.
- **Scale.** One GPU type, one model size, one replica. No tensor parallelism, no autoscaling under load, no speculative decoding.
- **Production traffic.** All prompts are synthetic, with fixed seeds, and every arm got the same prompts.
- **Repeat runs.** Each cell is one run. Cold starts are one observation each.

## How it works

```
config/poc.yaml          the POC spec: model, traffic shapes, concurrency levels, SLO, prices, variants, budget
providers/
  base.py                Endpoint and Provider interface: start, stop, cost
  modal_app.py           the Modal app: vLLM's OpenAI server on one GPU, weights in a Volume, boot timings in a Dict
  modal_vllm.py          deploy one app per variant, find its URL, stop it, and check that Modal lists it as stopped
  together.py            Together serverless: checks the model id against GET /v1/models before any paid request
bench/
  load.py                streaming client: closed loop, open loop, round trip, vLLM /metrics before and after each level
  prompts.py             seeded prompts: unique chat prompts, one shared long document, arithmetic questions
  stats.py               p50/p95 (p95 only at n >= 100, with a bootstrap interval), throughput, goodput
  cost.py                one spend ledger for the whole repo, with a hard stop at the budget
scripts/
  setup_modal.sh         one time: endpoint key, Modal secret, weights into the Volume (CPU only)
  run_poc.py             runs the phases: smoke, together, bf16, fp8, nocache
  report.py              results/<run_id>/summary.md, with sanity checks on the numbers
tests/                   the whole measurement path against a fake server with known timings; no network, no cost
results/                 committed on purpose: raw rows, per-level summaries, cold starts, spend
```

Each variant is its own Modal app (`inference-poc-<variant>`) with the settings baked into the image, so a run cannot mix settings. After every phase the runner stops the app and waits until Modal lists it as stopped. A running GPU costs money while nobody watches.

## Run it

```bash
uv sync
uv run pytest                                      # 11 tests, fake server, no cost

uv run modal setup                                 # links this machine to your Modal account
scripts/setup_modal.sh                             # endpoint key, secret, weights; no GPU time
uv run python scripts/run_poc.py --phases smoke    # one request, about $0.10
uv run python scripts/run_poc.py --phases bf16,fp8,nocache --run-id final
uv run python scripts/report.py final
```

For the Together arm, put `TOGETHER_API_KEY` in `.env` (see `.env.example`) and run `--phases together --run-id final`. The results go into the same report.

## What this run cost

$1.34 of Modal compute in total: the smoke test $0.08, bf16 $0.59, fp8 $0.49 and the no-cache arm $0.17 (including a first attempt that failed on a network timeout). These are upper bounds: container seconds from the first request to the stop. The budget stop in `config/poc.yaml` is $8.
