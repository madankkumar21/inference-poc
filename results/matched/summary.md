# Run matched

Self-hosted: vLLM on one Modal GPU; GPU, weights and vLLM version per variant in `config/poc.yaml`. together: `Qwen/Qwen3.5-9B` on its shared serverless fleet. together_dedicated: one replica of a Together dedicated endpoint, one GPU.
Goodput: share of measured requests with TTFT ≤ 1.0 s and end-to-end ≤ 15.0 s. p95 is shown only when n ≥ 100. Goodput uses client TTFT, so it includes the network path from the test machine; compare server TTFT for GPU-side latency.

Network round trip from the test machine (median of 7, warm connection): modal_vllm/h100-q35-fp8 111 ms; modal_vllm/h100-q35-fp8-mtp 113 ms; together_dedicated/h100-fp8 125 ms; modal_vllm/l40s-q35-bf16 231 ms; together_dedicated/h100-bf16 128 ms. Client TTFT includes this.

## Chat: closed loop, about 1,000 tokens in, 250 out

| Provider | Variant | Conc. | Run | n | Err | Prompt tok | Full len | TTFT p50 ms | TTFT p95 ms [95% CI] | Server TTFT mean ms | TPOT p50 ms | ITL p99 ms | Out tok/s | Out TPM | Req/s | Goodput | Prefix hit | $ per 1K req at 100% / 30% busy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_vllm | h100-q35-fp8 | 1 | 0 | 36 | 0 | 1,015 | 100% | 326 | n<100 | 45 | 5.4 | 63.8 | 151 | 9,049 | 0.60 | 100% | n/a | $2.434 / $8.112 |
| modal_vllm | h100-q35-fp8 | 4 | 0 | 132 | 0 | 1,017 | 100% | 364 | 435 [418, 441] | 60 | 5.6 | 64.4 | 565 | 33,902 | 2.26 | 100% | n/a | $0.650 / $2.165 |
| modal_vllm | h100-q35-fp8 | 16 | 0 | 432 | 0 | 1,016 | 100% | 540 | 656 [640, 669] | 153 | 6.6 | 70.3 | 1,834 | 110,061 | 7.34 | 100% | n/a | $0.200 / $0.667 |
| modal_vllm | h100-q35-fp8 | 64 | 0 | 960 | 0 | 1,016 | 100% | 1,059 | 1,614 [1,601, 1,623] | 446 | 11.4 | 156.3 | 4,114 | 246,857 | 16.46 | 47% | n/a | $0.089 / $0.297 |
| modal_vllm | h100-q35-fp8 | 1 | 1 | 36 | 0 | 1,015 | 100% | 343 | n<100 | 43 | 5.3 | 69.1 | 151 | 9,041 | 0.60 | 100% | n/a | $2.436 / $8.119 |
| modal_vllm | h100-q35-fp8 | 16 | 1 | 432 | 0 | 1,016 | 100% | 524 | 624 [620, 636] | 134 | 6.6 | 73.1 | 1,849 | 110,921 | 7.39 | 100% | n/a | $0.199 / $0.662 |
| modal_vllm | h100-q35-fp8-mtp | 1 | 0 | 38 | 0 | 1,015 | 100% | 331 | n<100 | 52 | 5.0 | 71.6 | 159 | 9,556 | 0.64 | 100% | n/a | $2.305 / $7.682 |
| modal_vllm | h100-q35-fp8-mtp | 4 | 0 | 136 | 0 | 1,017 | 100% | 373 | 499 [447, 625] | 77 | 5.5 | 70.5 | 569 | 34,158 | 2.28 | 100% | n/a | $0.645 / $2.149 |
| modal_vllm | h100-q35-fp8-mtp | 16 | 0 | 448 | 0 | 1,016 | 100% | 596 | 695 [684, 714] | 203 | 6.2 | 79.4 | 1,878 | 112,699 | 7.51 | 100% | n/a | $0.195 / $0.651 |
| modal_vllm | h100-q35-fp8-mtp | 64 | 0 | 960 | 0 | 1,016 | 100% | 1,162 | 1,700 [1,687, 1,718] | 540 | 10.8 | 171.4 | 4,161 | 249,668 | 16.64 | 39% | n/a | $0.088 / $0.294 |
| modal_vllm | h100-q35-fp8-mtp | 1 | 1 | 35 | 0 | 1,016 | 100% | 360 | n<100 | 52 | 5.0 | 80.2 | 149 | 8,923 | 0.59 | 97% | n/a | $2.468 / $8.227 |
| modal_vllm | h100-q35-fp8-mtp | 16 | 1 | 432 | 0 | 1,016 | 100% | 614 | 733 [701, 743] | 202 | 6.1 | 83.8 | 1,863 | 111,782 | 7.45 | 100% | n/a | $0.197 / $0.657 |
| together_dedicated | h100-fp8 | 1 | 0 | 35 | 0 | 1,016 | 100% | 258 | n<100 | n/a | 5.3 | 18.7 | 157 | 9,423 | 0.63 | 100% | n/a | $2.427 / $8.092 |
| together_dedicated | h100-fp8 | 4 | 0 | 140 | 0 | 1,017 | 100% | 318 | 361 [352, 373] | n/a | 5.5 | 19.4 | 586 | 35,136 | 2.34 | 100% | n/a | $0.651 / $2.170 |
| together_dedicated | h100-fp8 | 16 | 0 | 448 | 0 | 1,016 | 100% | 481 | 568 [564, 571] | n/a | 6.5 | 22.9 | 1,908 | 114,485 | 7.63 | 100% | n/a | $0.200 / $0.666 |
| together_dedicated | h100-fp8 | 64 | 0 | 1032 | 0 | 1,016 | 100% | 936 | 1,507 [1,470, 1,555] | n/a | 11.0 | 44.6 | 4,354 | 261,254 | 17.42 | 56% | n/a | $0.088 / $0.292 |
| together_dedicated | h100-fp8 | 1 | 1 | 37 | 0 | 1,015 | 100% | 262 | n<100 | n/a | 5.3 | 21.9 | 157 | 9,407 | 0.63 | 100% | n/a | $2.432 / $8.106 |
| together_dedicated | h100-fp8 | 16 | 1 | 448 | 0 | 1,016 | 100% | 497 | 579 [573, 585] | n/a | 6.5 | 22.8 | 1,906 | 114,334 | 7.62 | 100% | n/a | $0.200 / $0.667 |

## Natural prose: short questions, 250 tokens out (speculator-friendly text)

| Provider | Variant | Conc. | n | Err | Prompt tok | Full len | TTFT p50 ms | TTFT p95 ms [95% CI] | Server TTFT mean ms | TPOT p50 ms | ITL p99 ms | Out tok/s | Out TPM | Req/s | Goodput | Prefix hit | $ per 1K req at 100% / 30% busy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_vllm | h100-q35-fp8 | 1 | 40 | 0 | 39 | 40% | 331 | n<100 | 29 | 5.2 | 63.3 | 149 | 8,960 | 0.68 | 100% | n/a | $2.168 / $7.226 |
| modal_vllm | h100-q35-fp8 | 16 | 490 | 0 | 40 | 80% | 380 | 444 [438, 454] | 41 | 6.4 | 67.5 | 1,987 | 119,235 | 8.20 | 100% | n/a | $0.179 / $0.597 |
| modal_vllm | h100-q35-fp8-mtp | 1 | 40 | 0 | 39 | 50% | 332 | n<100 | 36 | 5.0 | 69.8 | 153 | 9,178 | 0.68 | 100% | n/a | $2.170 / $7.233 |
| modal_vllm | h100-q35-fp8-mtp | 16 | 479 | 0 | 40 | 81% | 416 | 495 [489, 507] | 74 | 6.4 | 73.5 | 1,969 | 118,156 | 8.13 | 100% | n/a | $0.181 / $0.602 |
| together_dedicated | h100-fp8 | 1 | 41 | 0 | 39 | 44% | 250 | n<100 | n/a | 5.3 | 20.1 | 157 | 9,401 | 0.68 | 100% | n/a | $2.240 / $7.466 |
| together_dedicated | h100-fp8 | 16 | 481 | 0 | 40 | 79% | 300 | 353 [339, 366] | n/a | 6.9 | 34.0 | 1,957 | 117,400 | 8.11 | 100% | n/a | $0.188 / $0.627 |

## Long context: one shared document, about 8,000 tokens in, 250 out

| Provider | Variant | Conc. | n | Err | Prompt tok | Full len | TTFT p50 ms | TTFT p95 ms [95% CI] | Server TTFT mean ms | TPOT p50 ms | ITL p99 ms | Out tok/s | Out TPM | Req/s | Goodput | Prefix hit | $ per 1K req at 100% / 30% busy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| modal_vllm | h100-q35-fp8 | 4 | 100 | 0 | 7,861 | 100% | 729 | 937 [914, 965] | 362 | 6.7 | 70.3 | 419 | 25,150 | 1.68 | 100% | n/a | $0.876 / $2.919 |
| modal_vllm | h100-q35-fp8-mtp | 4 | 104 | 0 | 7,861 | 100% | 740 | 949 [926, 995] | 391 | 6.2 | 94.4 | 435 | 26,101 | 1.74 | 98% | n/a | $0.844 / $2.812 |
| together_dedicated | h100-fp8 | 4 | 128 | 0 | 7,861 | 100% | 429 | 502 [466, 507] | n/a | 5.6 | 21.7 | 547 | 32,792 | 2.19 | 100% | n/a | $0.698 / $2.325 |

## Quality check: 200 arithmetic questions, temperature 0

| Provider | Variant | Exact |
|---|---|---|
| modal_vllm | h100-q35-fp8 | 103/200 |
| modal_vllm | h100-q35-fp8-mtp | 104/200 |
| together_dedicated | h100-fp8 | 102/200 |
| modal_vllm | l40s-q35-bf16 | 105/200 |
| together_dedicated | h100-bf16 | 106/200 |

- modal_vllm/h100-q35-fp8 gives the same answer as modal_vllm/l40s-q35-bf16 on 175/200 questions.
- modal_vllm/h100-q35-fp8-mtp gives the same answer as modal_vllm/l40s-q35-bf16 on 170/200 questions.
- together_dedicated/h100-fp8 gives the same answer as modal_vllm/l40s-q35-bf16 on 176/200 questions.
- together_dedicated/h100-bf16 gives the same answer as modal_vllm/l40s-q35-bf16 on 181/200 questions.

Multiplication answered with the number only is hard for a small model, so exact scores are low on every arm. A test this small cannot separate small quality changes; agreement with the full-precision reference is the more direct check on quantization.

## Cold starts (one observation each)

| Label | GPU | vLLM | Total to first token s | Schedule + boot s | Weights + engine s | First request TTFT ms |
|---|---|---|---|---|---|---|
| h100-q35-fp8-1 | NVIDIA H100 80GB HBM3 | 0.18.1 | 355.2 | 4.3 | 346.8 | 1,684 |
| h100-q35-fp8-mtp-1 | NVIDIA H100 80GB HBM3 | 0.18.1 | 390.0 | 8.0 | 380.0 | 1,886 |
| h100-fp8-create | Together H100 | n/a | 310.3 (request to READY) | n/a | n/a | n/a |
| l40s-q35-bf16-1 | NVIDIA L40S | 0.18.1 | 267.1 | 5.8 | 203.0 | 57,555 |
| h100-bf16-create | Together H100 | n/a | 156.8 (request to READY) | n/a | n/a | n/a |

The boot split subtracts this machine's clock from the container's clock, so it carries any clock difference between them. The second bf16 start has a warm image cache and a warm compile cache, so it shows both effects together.

## Sanity checks

- modal_vllm/h100-q35-fp8 c=64: client TTFT mean 1,051 ms vs server 446 ms; the gap (605 ms) is not explained by the 111 ms round trip.
- modal_vllm/h100-q35-fp8-mtp c=64: client TTFT mean 1,150 ms vs server 540 ms; the gap (610 ms) is not explained by the 113 ms round trip.

## Spend for this run

| Phase | USD | Basis |
|---|---|---|
| m1 | 2.215 | container-seconds, upper bound |
| m1 | 1.369 | container-seconds, upper bound |
| m2 | 1.422 | container-seconds, upper bound |
| t1 | 1.401 | GPU-seconds from create request to stop, upper bound |
| ref | 0.248 | container-seconds, upper bound |
| tref | 0.341 | GPU-seconds from create request to stop, upper bound |
| **Total** | **7.00** | |

How to read cost: a per-GPU arm's cost per 1,000 requests is its hourly price divided by the measured request rate, at the share of time the GPU is busy. Together dedicated uses its posted price per GPU hour, so it is priced the same way as the Modal containers.
