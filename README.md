# inference-poc

A 48-hour inference proof of concept (POC), run the way a solutions architect runs one for a customer. It serves an open-weight model on self-hosted vLLM, sweeps load, applies one tuning change at a time, and prices the result per 1,000 requests. The benchmark code only calls an OpenAI-compatible endpoint, so any provider fits. I ran it on [Modal](https://modal.com) with one NVIDIA L40S. I then pointed the same harness at Together's serverless API as a reference point. Last, I ran a matched comparison: the same model on one H100, on Modal with vLLM and on a Together dedicated endpoint.

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
- **Price at the real busy share, not at 100%.** One L40S running fp8 at 16 concurrent requests costs $0.24 per 1,000 requests when kept busy, and $0.81 at 30% busy. Together's serverless price for a small Qwen model came to $0.235 per 1,000 requests at the same token counts, at any load. So at the load where my GPU still met the latency target, self-hosting did not beat the per-token price even at 100% busy. It gets cheaper only at loads where no request meets the target.
- **Pick the operating point from the latency target, then read the cost.** At a 1 s TTFT target, fp8 kept 99% of requests inside it at 16 concurrent requests and 43% at 32. The cheapest row in the table is at 128, where no request met the target.
- **Plan for a minute of cold start.** Keep one warm container for interactive traffic, or accept about 50–95 s on the first request after scale to zero.

## Throughput against latency

More requests at once give more total output and a slower experience for each request. The fp8 arm shows the whole curve (same prompts, 60 s per level):

| Concurrent requests | Total output tok/s | Time per output token, p50 | Server TTFT, mean |
|---|---|---|---|
| 1 | 59 | 14.2 ms | 54 ms |
| 4 | 251 | 14.8 ms | 135 ms |
| 16 | 803 | 17.7 ms | 431 ms |
| 32 | 1,205 | 22.1 ms | 837 ms |
| 64 | 1,621 | 31.2 ms | 1,598 ms |
| 128 | 2,082 | 47.5 ms | 3,429 ms |

From 1 to 128 concurrent requests, total throughput rises 35x. Each request streams about 3.3x slower, and the wait for the first token grows from 54 ms to 3.4 s. The GPU batches the requests together, so each step serves more requests and takes longer.

So there is no single "performance" number. The latency target picks the operating point, and the throughput at that point sets the cost. With a 1 s TTFT target, fp8 kept 99% of requests inside the target at 16 concurrent requests and 43% at 32. Past that point, more load buys throughput that no user can use.

## The same harness against Together serverless

Same prompts, same load levels, same client machine. The model is different: Together no longer serves Qwen2.5-7B on serverless, so this arm is `Qwen/Qwen3.5-9B` (listed as an FP8 build), with its thinking mode turned off. Full rows are in [results/final/summary.md](results/final/summary.md).

| Concurrent requests | Output tok/s seen by this client | TTFT p50 | TTFT p95 | Time per output token | Failed requests |
|---|---|---|---|---|---|
| 1 | 128 | 256 ms | n<100 | 6.8 ms | 0 of 30 |
| 4 | 545 | 268 ms | 364 ms | 6.0 ms | 0 of 129 |
| 16 | 1,937 | 285 ms | 382 ms | 6.5 ms | 2 of 471 |
| 32 | 3,911 | 310 ms | 403 ms | 6.4 ms | 0 of 929 |
| 64 | 7,102 | 317 ms | 436 ms | 7.2 ms | 0 of 1,688 |

- **First-token time stayed flat as load rose.** TTFT p50 went from 256 ms to 317 ms between 1 and 64 concurrent requests. On my one L40S, server TTFT went from 54 ms to 1,598 ms over the same range. The network round trip to Together was 134 ms, so about half of that 256 ms is network.
- **Each request streamed at about 150 tokens per second** (6.0–7.2 ms per token) at every level. My L40S on fp8 streamed at 70 tokens per second with one request and 32 with 64.
- **Random arrivals changed little.** At 14.2 and 22.7 requests per second with Poisson arrivals, TTFT p95 was 414 ms and 393 ms.
- **Long shared document:** 362 ms TTFT p50 on an 8,000-token prompt at 4 concurrent requests.
- **Errors:** 5 of about 5,700 requests failed (four HTTP 503, one HTTP 500). I did not retry them.
- **Cost:** $0.235 per 1,000 chat requests at the posted price ($0.17 per million input tokens, $0.25 per million output), and $1.40 per 1,000 for the 8,000-token prompts.

How to read this, and how not to:
- **This is not a hardware comparison.** The Together arm is a shared fleet of GPUs I can't see, and a newer, slightly larger model. The throughput column is what one client received, not what one GPU can do. The fair reading is "what a per-token API gave me with no setup", next to "what one rented GPU gave me after two days of work".
- **The flat TTFT is the point.** A single GPU trades latency for throughput, as the section above shows. A fleet with spare capacity does not make one customer choose, up to that customer's rate limit. I did not find where that limit is: 64 concurrent requests and 22.7 requests per second were still flat.
- **The first attempt was less clean.** In my first run, a few requests at 16 or more concurrent waited 30–55 s for the first token and then streamed at normal speed. One request took 145 s end to end. A second full run later the same day had none: the slowest request took 12 s. I can't tell the cause from the client side; a new account's rate limit is one candidate. The first attempt's rows are in [results/together-attempt1/](results/together-attempt1/).
- **Quality:** 25 of 50 on the arithmetic check, against 21 of 50 for my bf16 arm. It is a different model, so this says nothing about quantization.

## Matched: the same model on one H100

The serverless arm above is not a fair comparison, so I ran a matched one. Every arm uses Qwen3.5-9B with thinking off, one H100, one replica, and the same harness, prompts and client. Each provider uses its own FP8 build of the model. Full rows: [results/matched/summary.md](results/matched/summary.md).

- **M1:** Modal, vLLM 0.18.1, [RedHatAI/Qwen3.5-9B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3.5-9B-FP8-dynamic), no speculative decoding.
- **M2:** M1 plus vLLM's MTP speculative decoding. MTP (multi-token prediction) uses a draft layer that ships with Qwen3.5 to propose the next token.
- **T1:** a Together [dedicated endpoint](https://docs.together.ai/docs/dedicated-inference) with the 1x H100 FP8 configuration for this model. On Together, speculative decoding is a property of the configuration you pick ([configs](https://docs.together.ai/docs/dedicated-endpoints/configs)). On 2026-11-16 no configuration for this model had a speculator, so T1 runs without one, like M1.

Chat traffic, about 1,000 tokens in and 250 out. Each cell is M1 / M2 / T1.

| Concurrent requests | Time per output token p50 | Output tok/s | TTFT p50 | Gap between tokens p99 | Goodput |
|---|---|---|---|---|---|
| 1 | 5.4 / 5.0 / 5.3 ms | 151 / 159 / 157 | 326 / 331 / 258 ms | 64 / 72 / 19 ms | 100 / 100 / 100% |
| 4 | 5.6 / 5.5 / 5.5 ms | 565 / 569 / 586 | 364 / 373 / 318 ms | 64 / 71 / 19 ms | 100 / 100 / 100% |
| 16 | 6.6 / 6.2 / 6.5 ms | 1,834 / 1,878 / 1,908 | 540 / 596 / 481 ms | 70 / 79 / 23 ms | 100 / 100 / 100% |
| 64 | 11.4 / 10.8 / 11.0 ms | 4,114 / 4,161 / 4,354 | 1,059 / 1,162 / 936 ms | 156 / 171 / 45 ms | 47 / 39 / 56% |
| 4, on an 8,000-token document | 6.7 / 6.2 / 5.6 ms | 419 / 435 / 547 | 729 / 740 / 429 ms | 70 / 94 / 22 ms | 100 / 98 / 100% |

What the matched run shows:
- **At low and medium load, decode speed is the same within about 5%.** For a 9B model on one H100, the GPU's memory bandwidth sets the speed of each token, whoever runs the stack.
- **Together pulled ahead where the work gets harder.** At 64 concurrent requests it gave 6% more output and 56% goodput against 47%. On the 8,000-token document it gave 31% more output (547 against 419 tok/s), and TTFT p50 was 429 ms against 729 ms. I cannot say which part of Together's stack causes this: the engine, kernels, FP8 build, scheduler and host machine all differ.
- **Tokens arrived more evenly from Together.** The 99th-percentile gap between streamed tokens was 19–45 ms on Together and 64–171 ms on Modal. Modal's web proxy may group stream chunks, so I do not credit this to the GPU.
- **vLLM's MTP accepted 72% of drafted tokens (1.72 tokens per step) but cut time per token by only 2–8%.** At 64 concurrent requests, M2's goodput fell to 39%. I did not profile why. A speculator trained for the customer's traffic could do better; this one ships with the model.
- **The repeats agreed.** I ran 1 and 16 concurrent requests twice on each arm. Output tok/s changed by 1% or less, except M2 at 1 concurrent request (159 against 149).
- **Read TTFT with care.** Client TTFT includes the network path. The round trip was 111 ms to Modal and 125 ms to Together's API. Together reported the replica in eu-west-1, and the Modal containers ran in us-east, yet Together's TTFT was lower. I did not find out how traffic reached the replica. That is why the table leads with time per token and output rate.

**Quality.** 200 arithmetic questions at temperature 0, each answered with the number only. Agreement means the same answer as Modal's bf16 reference on an L40S.

| Arm | Exact | Same answer as the bf16 reference |
|---|---|---|
| M1: Modal FP8 | 103/200 | 175/200 |
| M2: Modal FP8 with MTP | 104/200 | 170/200 |
| T1: Together FP8 | 102/200 | 176/200 |
| Together bf16, 1x H100 | 106/200 | 181/200 |
| Modal bf16, L40S (the reference) | 105/200 | n/a |

Two full-precision builds agreed on 181 of 200, and M1 and M2 agreed on 179 of 200, although speculative decoding is designed not to change answers. So small numeric differences under batching change about one answer in ten. Both FP8 builds fall inside that band. This test shows no quality loss from either FP8 build, and it is too small to prove there is none.

**Cost.** Together dedicated costs $5.49 per H100 hour ([pricing](https://www.together.ai/pricing)). The Modal container costs $5.29 per hour: $3.95 for the H100 plus 4 CPU cores and 32 GiB of memory ([pricing](https://modal.com/pricing)). At the same speed, cost per request is close to equal: $0.20 per 1,000 chat requests at 16 concurrent on both, at 100% busy. On the 8,000-token document, Together was cheaper per request because it was faster: $0.70 against $0.88 per 1,000.

**Startup.** Modal took 355 s (M1) and 390 s (M2) from the first request to the first token, on a warm image. Together took 310 s from the create request to a ready replica. One observation each.

**What this does not show.** One replica cannot show routing across replicas, autoscaling or cache-aware routing. I could not test Together's own speculators, because this model has none listed. The prompts are synthetic, each cell has one run plus the two repeats, and the quality check is arithmetic only.

## What I would hand to a managed platform

Running this myself showed what the work is. Each row is something I had to do or pay for by hand.

| What I found | What it costs when you self-host | What a managed inference platform takes on |
|---|---|---|
| Cold start took 51–95 s | Keep a GPU warm and pay for idle time, or make the first user wait | Serverless endpoints are already warm; dedicated endpoints stay up |
| One L40S on fp8 costs $0.24 per 1,000 requests at 100% busy and $0.81 at 30% | You have to predict traffic, and you pay for every idle second. The per-token price I measured was $0.235 at any load | Pay per token while traffic is low or spiky, then move to dedicated capacity when it is steady |
| FP8 gave 1.5–1.7x, and prefix caching gave 28x on first token | I found each gain by hand, one change per run. A new model or a new traffic shape means doing it again | The provider tunes the engine. Together describes tuning to the live traffic shape in its [Compiler post](https://www.together.ai/blog/together-compiler-continual-compilation-for-ai-inference) |
| FP8 changed 12 of 50 answers slightly | Every precision or engine change needs a quality check on your own prompts | The check is still yours to define. A provider can run it with you on each change |
| One GPU type, one 7B model | Larger models need GPUs that are hard to rent one at a time | A fleet of current GPUs and a catalog of models that are already deployed |

The choice is the usual build or buy question. Self-hosting can be cheaper for steady, high traffic when a team wants to own the tuning. For most teams the question is who maintains the serving stack after the first week. Prices: [Together pricing](https://www.together.ai/pricing).

## What I did not test

- **Together at scale.** The matched run is one dedicated replica. I did not find the serverless rate limit, and Together exposes no server-side TTFT, so its latency is client-side only.
- **Answer quality, properly.** 50 arithmetic questions with "number only" answers is too hard for a 7B model and too small to detect a small change. A real POC uses the customer's prompts and a task-specific grader.
- **The bf16 vs fp8 goodput gap.** Goodput (the share of requests that meet the latency target) uses client TTFT. The bf16 container had a much slower network path, so its goodput is lower for reasons that have nothing to do with FP8. The bf16 container also showed 290 ms gaps between some streamed tokens, against about 30 ms on fp8 at concurrency 1 and 4. This is probably the same network path, but I did not prove it.
- **Scale.** One replica per arm. No tensor parallelism and no autoscaling under load. Speculative decoding only as vLLM's MTP on the matched run.
- **Production traffic.** All prompts are synthetic, with fixed seeds, and every arm got the same prompts.
- **Repeat runs.** Each L40S cell is one run. The matched run repeats 1 and 16 concurrent requests. Cold starts are one observation each.

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
uv run pytest                                      # 22 tests, fake servers, no cost

uv run modal setup                                 # links this machine to your Modal account
scripts/setup_modal.sh                             # endpoint key, secret, weights; no GPU time
uv run python scripts/run_poc.py --phases smoke    # one request, about $0.10
uv run python scripts/run_poc.py --phases bf16,fp8,nocache --run-id final
uv run python scripts/report.py final
```

For the Together arm, put `TOGETHER_API_KEY` in `.env` (see `.env.example`) and run `--phases together --run-id final`. The results go into the same report.

The matched run needs a Together project API key in `.env`. A dedicated endpoint bills per minute until it is deleted, so every phase stops and deletes it on exit, and `scripts/stop_all.py` removes anything left over:

```bash
uv run python scripts/stop_all.py --check          # lists anything still running; exits 1 if so
uv run python scripts/run_poc.py --phases m1,m2,t1,ref,tref --run-id matched
uv run python scripts/report.py matched
```

## What this run cost

$10.49 logged in total, against a budget stop of $14 in `config/poc.yaml` ($8 before the matched run).

- **Modal: $1.34.** The smoke test $0.08, bf16 $0.59, fp8 $0.49 and the no-cache arm $0.17 (including a first attempt that failed on a network timeout). These are upper bounds: container seconds from the first request to the stop.
- **Together: $2.16** at the posted token price: $1.48 for the final run and $0.68 for the first attempt.
- **Matched run: $6.99.** M1 $1.37, plus $2.22 for a first M1 start that never came up (see below). M2 $1.42, Together FP8 $1.40, the L40S bf16 reference $0.25, and Together bf16 $0.34. These are upper bounds: Modal container seconds, and Together time from the create request to the stop.

## What went wrong, and what I changed

The first Together run exposed a bug in my own harness. The harness measured each load level from the moment every worker had finished its first request. At 64 concurrent requests, a few first requests stalled for the whole level, so the measured window shrank to almost zero and the request rate came out in the millions. The open-loop step takes its arrival rate from that number, so it started an uncontrolled burst of requests, and the process died before it saved any rows. The cost of that burst is not in the ledger above, because the ledger counts tokens from saved rows.

What I changed, each with an offline test:
- **The measured window** starts when 90% of workers have finished their first request (`measured_window` in `bench/load.py`).
- **The open-loop step refuses any rate above 100 requests per second,** and a level with a broken window is left out of the peak.
- **A request has a 60 s deadline** and counts as an error past it. The slowest self-hosted request took 28 s, so the earlier arms are unchanged.
- **The network probe for a hosted API** now uses a small response. The old probe downloaded Together's 436 KB model list, so it measured download time.

The lesson I would carry into a customer POC: a spend guard that checks after each step does not protect a step that never finishes. Cap the request rate before the first paid request.

The matched run had two setup failures of its own:
- **A tokenizer the engine could not load.** The FP8 checkpoint names a tokenizer class that only exists in transformers 5, and vLLM 0.18.1 requires transformers 4. The container crashed on start and restarted until my 25-minute health check gave up, at a cost of $2.22. The fix loads the tokenizer from the base `Qwen/Qwen3.5-9B` repo, which has the same tokenizer files. I checked the fix on a CPU-only container before the next GPU start. The health check now gives up after 15 minutes.
- **Arguments lost on the way into the container.** Modal's image environment step removed the backslashes in a JSON argument, so the container could not parse its own settings. The arguments now go in base64-encoded.

The lesson: check that the model loads on a cheap machine before paying for a GPU. A crash on start costs as much as a benchmark.
