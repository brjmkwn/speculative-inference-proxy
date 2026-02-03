import asyncio
import json
import time
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


async def run_single_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int,
    use_speculative: bool,
    speculative_k: int = 4
) -> Dict[str, Any]:
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "use_speculative": use_speculative,
        "speculative_k": speculative_k
    }

    start_time = time.perf_counter()
    first_token_time = None
    token_times = []
    total_tokens = 0
    full_text = []
    usage_info = {}

    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120.0
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(f"HTTP {response.status_code}: {text.decode('utf-8')}")

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):].strip()
                if data_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            token_times.append(now)
                            full_text.append(content)
                            total_tokens += 1
                    
                    if "usage" in chunk and chunk["usage"]:
                        usage_info = chunk["usage"]
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        return {"error": str(e), "success": False}

    end_time = time.perf_counter()
    total_duration = end_time - start_time
    ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else (total_duration * 1000.0)

    itls = []
    for i in range(1, len(token_times)):
        itls.append((token_times[i] - token_times[i - 1]) * 1000.0)

    tps = (total_tokens / total_duration) if total_duration > 0 else 0.0
    acceptance_rate = usage_info.get("acceptance_rate", 0.0)

    return {
        "success": True,
        "total_tokens": total_tokens,
        "duration_s": total_duration,
        "ttft_ms": ttft_ms,
        "itl_p50_ms": float(np.percentile(itls, 50)) if itls else 0.0,
        "itl_p95_ms": float(np.percentile(itls, 95)) if itls else 0.0,
        "tokens_per_sec": tps,
        "acceptance_rate": acceptance_rate,
        "text_length": len("".join(full_text))
    }


async def run_ab_benchmark(
    base_url: str = "http://localhost:8000",
    prompts_file: str = "benchmarks/test_prompts.json",
    max_tokens: int = 256,
    iterations: int = 5,
    speculative_k: int = 4
):
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        print(f"Error: Prompts file not found at {prompts_file}")
        return

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Running A/B benchmark (url={base_url}, K={speculative_k}, runs={iterations}, prompts={len(prompts)})")

    results_summary = []

    async with httpx.AsyncClient() as client:
        try:
            health_resp = await client.get(f"{base_url}/health", timeout=10.0)
            health_data = health_resp.json()
            print(f"Server ready (device={health_data.get('device')})")
        except Exception as e:
            print(f"Warning: Health check failed ({e})")

        for p_idx, prompt_item in enumerate(prompts):
            p_id = prompt_item["id"]
            p_text = prompt_item["prompt"]
            p_cat = prompt_item.get("category", "general")

            print(f"[{p_idx + 1}/{len(prompts)}] Testing {p_id} ({p_cat})...", end="", flush=True)

            vanilla_metrics: List[Dict[str, Any]] = []
            speculative_metrics: List[Dict[str, Any]] = []

            for it in range(iterations):
                res = await run_single_request(client, base_url, p_text, max_tokens, use_speculative=False)
                if res.get("success"):
                    vanilla_metrics.append(res)

            for it in range(iterations):
                res = await run_single_request(client, base_url, p_text, max_tokens, use_speculative=True, speculative_k=speculative_k)
                if res.get("success"):
                    speculative_metrics.append(res)

            if not vanilla_metrics or not speculative_metrics:
                print(" failed")
                continue

            v_tps = float(np.mean([m["tokens_per_sec"] for m in vanilla_metrics]))
            s_tps = float(np.mean([m["tokens_per_sec"] for m in speculative_metrics]))
            speedup = s_tps / max(v_tps, 1e-6)

            v_ttft = float(np.mean([m["ttft_ms"] for m in vanilla_metrics]))
            s_ttft = float(np.mean([m["ttft_ms"] for m in speculative_metrics]))

            v_itl = float(np.mean([m["itl_p50_ms"] for m in vanilla_metrics]))
            s_itl = float(np.mean([m["itl_p50_ms"] for m in speculative_metrics]))

            acc_rate = float(np.mean([m["acceptance_rate"] for m in speculative_metrics]))

            print(f" speedup: {speedup:.2f}x (baseline: {v_tps:.1f} t/s, speculative: {s_tps:.1f} t/s, alpha: {acc_rate * 100:.1f}%)")

            results_summary.append({
                "prompt_id": p_id,
                "category": p_cat,
                "vanilla_tps": v_tps,
                "speculative_tps": s_tps,
                "speedup": speedup,
                "vanilla_itl_ms": v_itl,
                "speculative_itl_ms": s_itl,
                "vanilla_ttft_ms": v_ttft,
                "speculative_ttft_ms": s_ttft,
                "acceptance_rate": acc_rate
            })

    md_table = generate_markdown_report(results_summary)
    print("\n" + md_table)

    bench_file = Path("BENCHMARKS.md")
    if bench_file.exists():
        content = bench_file.read_text(encoding="utf-8")
        if "<!-- BENCHMARK_RESULTS_START -->" in content:
            parts = content.split("<!-- BENCHMARK_RESULTS_START -->")
            rest = parts[1].split("<!-- BENCHMARK_RESULTS_END -->")
            new_content = parts[0] + "<!-- BENCHMARK_RESULTS_START -->\n" + md_table + "\n<!-- BENCHMARK_RESULTS_END -->" + rest[1]
            bench_file.write_text(new_content, encoding="utf-8")
            print("Updated BENCHMARKS.md")


def generate_markdown_report(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No benchmark results collected."

    lines = [
        "| Prompt ID | Category | Baseline TPS | Speculative TPS | Speedup | Baseline ITL | Speculative ITL | Draft Acceptance (α) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    avg_speedup = np.mean([r["speedup"] for r in results])
    avg_alpha = np.mean([r["acceptance_rate"] for r in results])

    for r in results:
        lines.append(
            f"| `{r['prompt_id']}` | {r['category']} | {r['vanilla_tps']:.1f} t/s | **{r['speculative_tps']:.1f} t/s** | **{r['speedup']:.2f}x** | {r['vanilla_itl_ms']:.1f} ms | {r['speculative_itl_ms']:.1f} ms | {r['acceptance_rate'] * 100:.1f}% |"
        )

    lines.append(
        f"| **OVERALL AVERAGE** | **All** | — | — | **{avg_speedup:.2f}x** | — | — | **{avg_alpha * 100:.1f}%** |"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative decoding A/B benchmark runner")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of proxy server")
    parser.add_argument("--prompts", default="benchmarks/test_prompts.json", help="Path to prompts JSON")
    parser.add_argument("--tokens", type=int, default=256, help="Max tokens per prompt")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per prompt")
    parser.add_argument("--k", type=int, default=4, help="Lookahead speculative tokens K")

    args = parser.parse_args()
    asyncio.run(run_ab_benchmark(
        base_url=args.url,
        prompts_file=args.prompts,
        max_tokens=args.tokens,
        iterations=args.iterations,
        speculative_k=args.k
    ))
