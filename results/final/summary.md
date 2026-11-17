# Run final

Self-hosted: vLLM on one Modal GPU; GPU, weights and vLLM version per variant in `config/poc.yaml`. together: `Qwen/Qwen3.5-9B` on its shared serverless fleet. together_dedicated: one replica of a Together dedicated endpoint, one GPU.
Goodput: share of measured requests with TTFT ≤ 1.0 s and end-to-end ≤ 15.0 s. p95 is shown only when n ≥ 100. Goodput uses client TTFT, so it includes the network path from the test machine; compare server TTFT for GPU-side latency.

Network round trip from the test machine (median of 7, warm connection): modal_vllm/bf16 327 ms; modal_vllm/fp8 59 ms; modal_vllm/bf16-nocache 39 ms; together/serverless 134 ms. Client TTFT includes this.

## Chat: closed loop, about 1,000 tokens in, 250 out

| Provider | Variant | Conc. | n | Err | Prompt tok | Full len | TTFT p50 ms | TTFT p95 ms [95% CI] | Server TTFT mean ms | TPOT p50 ms | ITL p99 ms | Out tok/s | Out TPM | Req/s | Goodput | Prefix hit | $ per 1K req at 100% / 30% busy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_vllm | bf16 | 1 | 9 | 0 | 1,031 | 100% | 976 | n<100 | 73 | 21.1 | 291.7 | 40 | 2,403 | 0.16 | 89% | 2% | $4.875 / $16.249 |
| modal_vllm | bf16 | 4 | 32 | 0 | 1,033 | 100% | 1,032 | n<100 | 128 | 23.3 | 282.6 | 148 | 8,850 | 0.59 | 44% | 2% | $1.324 / $4.412 |
| modal_vllm | bf16 | 16 | 112 | 0 | 1,032 | 100% | 1,319 | 1,720 [1,606, 1,833] | 364 | 26.6 | 284.4 | 508 | 30,462 | 2.03 | 17% | 2% | $0.385 / $1.282 |
| modal_vllm | bf16 | 32 | 164 | 0 | 1,032 | 100% | 1,726 | 2,617 [2,551, 2,700] | 866 | 33.4 | 281.0 | 740 | 44,379 | 2.96 | 12% | 2% | $0.264 / $0.880 |
| modal_vllm | bf16 | 64 | 256 | 0 | 1,032 | 100% | 2,880 | 4,698 [4,583, 4,788] | 2,000 | 42.8 | 284.2 | 1,189 | 71,341 | 4.76 | 5% | 2% | $0.164 / $0.547 |
| modal_vllm | bf16 | 128 | 256 | 0 | 1,032 | 100% | 4,209 | 4,585 [4,522, 4,626] | 3,368 | 65.9 | 294.4 | 1,803 | 108,177 | 7.21 | 0% | 2% | $0.108 / $0.361 |
| modal_vllm | fp8 | 1 | 14 | 0 | 1,031 | 100% | 169 | n<100 | 54 | 14.2 | 33.6 | 59 | 3,518 | 0.23 | 86% | 2% | $3.330 / $11.099 |
| modal_vllm | fp8 | 4 | 60 | 0 | 1,032 | 100% | 273 | n<100 | 135 | 14.8 | 29.1 | 251 | 15,042 | 1.00 | 100% | 2% | $0.779 / $2.596 |
| modal_vllm | fp8 | 16 | 192 | 0 | 1,032 | 100% | 602 | 964 [936, 979] | 431 | 17.7 | 116.6 | 803 | 48,164 | 3.21 | 99% | 2% | $0.243 / $0.811 |
| modal_vllm | fp8 | 32 | 288 | 0 | 1,032 | 100% | 1,132 | 1,954 [1,884, 1,974] | 837 | 22.1 | 316.0 | 1,205 | 72,321 | 4.82 | 43% | 2% | $0.162 / $0.540 |
| modal_vllm | fp8 | 64 | 384 | 0 | 1,032 | 100% | 1,847 | 3,804 [3,694, 3,921] | 1,598 | 31.2 | 358.6 | 1,621 | 97,283 | 6.49 | 14% | 2% | $0.120 / $0.401 |
| modal_vllm | fp8 | 128 | 384 | 0 | 1,032 | 100% | 4,460 | 6,949 [6,647, 7,065] | 3,429 | 47.5 | 365.2 | 2,082 | 124,894 | 8.33 | 0% | 2% | $0.094 / $0.313 |
| together | serverless | 1 | 30 | 0 | 1,015 | 100% | 256 | n<100 | n/a | 6.8 | 72.6 | 128 | 7,694 | 0.51 | 100% | n/a | $0.235 / $0.235 (posted price) |
| together | serverless | 4 | 129 | 0 | 1,017 | 100% | 268 | 364 [339, 392] | n/a | 6.0 | 72.7 | 545 | 32,721 | 2.18 | 100% | n/a | $0.235 / $0.235 (posted price) |
| together | serverless | 16 | 471 | 2 | 1,016 | 100% | 285 | 382 [368, 394] | n/a | 6.5 | 74.1 | 1,937 | 116,223 | 7.75 | 100% | n/a | $0.235 / $0.235 (posted price) |
| together | serverless | 32 | 929 | 0 | 1,016 | 100% | 310 | 403 [392, 422] | n/a | 6.4 | 76.0 | 3,911 | 234,682 | 15.65 | 99% | n/a | $0.235 / $0.235 (posted price) |
| together | serverless | 64 | 1688 | 0 | 1,016 | 100% | 317 | 436 [422, 448] | n/a | 7.2 | 77.4 | 7,102 | 426,109 | 28.41 | 99% | n/a | $0.235 / $0.235 (posted price) |

## Long context: one shared document, about 8,000 tokens in, 250 out

| Provider | Variant | Conc. | n | Err | Prompt tok | Full len | TTFT p50 ms | TTFT p95 ms [95% CI] | Server TTFT mean ms | TPOT p50 ms | ITL p99 ms | Out tok/s | Out TPM | Req/s | Goodput | Prefix hit | $ per 1K req at 100% / 30% busy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_vllm | bf16 | 4 | 36 | 0 | 7,877 | 100% | 887 | n<100 | 44 | 22.6 | 232.4 | 151 | 9,052 | 0.60 | 78% | 100% | $1.294 / $4.314 |
| modal_vllm | bf16-nocache | 4 | 27 | 0 | 7,877 | 100% | 1,346 | n<100 | 1,238 | 27.9 | 160.1 | 116 | 6,987 | 0.47 | 26% | n/a | $1.677 / $5.589 |
| together | serverless | 4 | 122 | 0 | 7,861 | 100% | 362 | 444 [430, 500] | n/a | 6.1 | 72.3 | 516 | 30,939 | 2.06 | 100% | n/a | $1.399 / $1.399 (posted price) |

## Open loop: Poisson arrivals at a share of each arm's closed-loop peak

| Provider | Variant | Load | Rate req/s | n | Err | Prompt tok | Full len | TTFT p50 ms | TTFT p95 ms [95% CI] | Server TTFT mean ms | TPOT p50 ms | ITL p99 ms | Out tok/s | Out TPM | Req/s | Goodput | Prefix hit | $ per 1K req at 100% / 30% busy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_vllm | bf16 | 0.5 | 3.606 | 229 | 0 | 1,032 | 100% | 1,048 | 1,316 [1,256, 1,393] | 173 | 39.8 | 243.8 | 846 | 50,754 | 3.38 | 41% | 2% | $0.231 / $0.769 |
| modal_vllm | bf16 | 0.8 | 5.769 | 347 | 0 | 1,032 | 100% | 1,230 | 2,868 [2,497, 3,012] | 284 | 55.3 | 295.1 | 1,218 | 73,053 | 4.87 | 8% | 2% | $0.160 / $0.534 |
| modal_vllm | fp8 | 0.5 | 4.163 | 260 | 0 | 1,032 | 100% | 289 | 602 [551, 647] | 111 | 22.9 | 177.1 | 1,020 | 61,213 | 4.08 | 100% | 2% | $0.191 / $0.638 |
| modal_vllm | fp8 | 0.8 | 6.661 | 393 | 0 | 1,032 | 100% | 596 | 1,312 [1,224, 1,403] | 227 | 37.0 | 297.7 | 1,509 | 90,551 | 6.04 | 82% | 2% | $0.129 / $0.431 |
| together | serverless | 0.5 | 14.204 | 851 | 2 | 1,016 | 100% | 295 | 414 [396, 436] | n/a | 6.6 | 75.7 | 3,385 | 203,072 | 13.54 | 100% | n/a | $0.235 / $0.235 (posted price) |
| together | serverless | 0.8 | 22.726 | 1334 | 1 | 1,016 | 100% | 302 | 393 [388, 402] | n/a | 6.5 | 76.6 | 5,337 | 320,204 | 21.35 | 100% | n/a | $0.235 / $0.235 (posted price) |

## Quality check: 50 arithmetic questions, temperature 0

| Provider | Variant | Exact |
|---|---|---|
| modal_vllm | bf16 | 21/50 |
| modal_vllm | fp8 | 19/50 |
| together | serverless | 25/50 |

- modal_vllm/fp8 gives the same answer as modal_vllm/bf16 on 38/50 questions.
- together/serverless gives the same answer as modal_vllm/bf16 on 10/50 questions.

Multiplication answered with the number only is hard for a small model, so exact scores are low on every arm. A test this small cannot separate small quality changes; agreement with the full-precision reference is the more direct check on quantization.

## Cold starts (one observation each)

| Label | GPU | vLLM | Total to first token s | Schedule + boot s | Weights + engine s | First request TTFT ms |
|---|---|---|---|---|---|---|
| bf16-1 | NVIDIA L40S | 0.11.0 | 67.8 | 8.9 | 56.0 | 2,910 |
| bf16-2-warm-caches | NVIDIA L40S | 0.11.0 | 50.9 | 4.6 | 43.2 | 2,774 |
| fp8-1 | NVIDIA L40S | 0.11.0 | 94.8 | 3.8 | 90.1 | 975 |
| bf16-nocache-1 | NVIDIA L40S | 0.11.0 | 58.7 | 3.8 | 54.9 | 618 |
| bf16-nocache-1 | NVIDIA L40S | 0.11.0 | 54.3 | 8.0 | 45.7 | 637 |

The boot split subtracts this machine's clock from the container's clock, so it carries any clock difference between them. The second bf16 start has a warm image cache and a warm compile cache, so it shows both effects together.

## Sanity checks

- modal_vllm/fp8 c=1: client TTFT mean 572 ms vs server 54 ms; the gap (518 ms) is not explained by the 59 ms round trip.
- modal_vllm/fp8 c=32: client TTFT mean 1,134 ms vs server 837 ms; the gap (297 ms) is not explained by the 59 ms round trip.
- modal_vllm/fp8 c=64: client TTFT mean 2,103 ms vs server 1,598 ms; the gap (505 ms) is not explained by the 59 ms round trip.
- modal_vllm/fp8 c=128: client TTFT mean 4,273 ms vs server 3,429 ms; the gap (844 ms) is not explained by the 59 ms round trip.
- together/serverless: 2 failed requests in the chat sweep; samples: []

## Spend for this run

| Phase | USD | Basis |
|---|---|---|
| bf16 | 0.542 | first container, container-seconds upper bound |
| bf16 | 0.050 | container-seconds, upper bound |
| fp8 | 0.491 | container-seconds, upper bound |
| nocache | 0.071 | container-seconds, upper bound |
| nocache | 0.102 | container-seconds, upper bound |
| together | 0.679 | first attempt 2026-11-16, tokens at posted price from saved rows; excludes an open-loop burst that saved no rows (harness bug) |
| together | 1.480 | tokens at posted price (sweeps only) |
| **Total** | **3.41** | |

How to read cost: a per-GPU arm's cost per 1,000 requests is its hourly price divided by the measured request rate, at the share of time the GPU is busy. Together serverless uses its posted input and output price at the measured token counts, and does not depend on load. Its throughput is what this client saw on a shared fleet with rate limits, not the capacity of one GPU.
