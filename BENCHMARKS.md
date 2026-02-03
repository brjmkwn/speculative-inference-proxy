# Benchmark Results and Sizing Guide

Performance evaluation comparing standard target model autoregression (Baseline) against speculative decoding across different tasks and lookahead window sizes.

## Experimental Setup

- **Hardware:** NVIDIA A100-SXM4 (80GB VRAM)
- **Target Model:** `Qwen/Qwen2.5-7B-Instruct` (bfloat16)
- **Draft Model:** `Qwen/Qwen2.5-0.5B-Instruct` (bfloat16)
- **Lookahead Window (K):** 4
- **Sampling:** Temperature = 0.0 (Greedy argmax verification)

## Results

<!-- BENCHMARK_RESULTS_START -->
| Prompt ID | Category | Baseline TPS | Speculative TPS | Speedup | Baseline ITL | Speculative ITL | Draft Acceptance (α) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `code-binary-search-tree` | coding | 34.2 t/s | 89.5 t/s | 2.62x | 29.2 ms | 11.2 ms | 78.4% |
| `json-schema-extraction` | json_schema | 35.8 t/s | 104.2 t/s | 2.91x | 27.9 ms | 9.6 ms | 84.1% |
| `reasoning-math-proof` | reasoning | 33.5 t/s | 76.8 t/s | 2.29x | 29.8 ms | 13.0 ms | 68.5% |
| `sql-query-optimization` | database | 36.1 t/s | 98.4 t/s | 2.73x | 27.7 ms | 10.2 ms | 81.2% |
| `system-design-rate-limiter` | architecture | 34.0 t/s | 82.3 t/s | 2.42x | 29.4 ms | 12.1 ms | 72.8% |
| `code-async-thread-pool` | coding | 34.7 t/s | 93.1 t/s | 2.68x | 28.8 ms | 10.7 ms | 79.6% |
| **OVERALL AVERAGE** | **All** | **34.7 t/s** | **90.7 t/s** | **2.61x** | **28.8 ms** | **11.1 ms** | **77.4%** |
<!-- BENCHMARK_RESULTS_END -->

## Lookahead Window Sizing

The optimal choice of $K$ depends on the acceptance rate of the workload and the draft model's overhead relative to the target model:

| K | Workload Profile | Expected Speedup | Notes |
|---|---|---|---|
| 2 | High temperature (> 0.8), open-ended generation | 1.4x - 1.7x | Lower wasted compute on draft rejection |
| 4 | Structured extraction, code, reasoning | 2.3x - 3.0x | Balanced default for most production tasks |
| 6 | Strict schema JSON, high prompt prefix caching | 2.6x - 3.4x | High draft acceptance compensates for longer sequential draft passes |
| >= 8 | Diminishing returns | 2.0x - 2.8x | Sequential draft latency starts bottlenecking overall throughput |

## Running Benchmarks

To reproduce these benchmarks locally:

```bash
python benchmarks/benchmark_runner.py \
  --url http://localhost:8000 \
  --prompts benchmarks/test_prompts.json \
  --tokens 256 \
  --iterations 3 \
  --k 4
```
