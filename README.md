# Speculative Decoding Inference Proxy

An OpenAI-compatible inference proxy that implements speculative decoding across draft and target models.

## Overview

Speculative decoding speeds up autoregressive LLM decoding by having a small, fast draft model generate candidate token sequences and verifying them in parallel with a single forward pass of the target model. 

```
+----------------+       POST /v1/chat/completions       +-----------------------+
|  Client / SDK  | ------------------------------------> |  FastAPI Gateway      |
+----------------+                                       +-----------------------+
                                                                     |
                                             +-----------------------+-----------------------+
                                             |                                               |
                                             v                                               v
                                  +--------------------+                          +--------------------+
                                  |    Draft Model     |                          |    Target Model    |
                                  | (e.g. Qwen2.5-0.5B)|                          | (e.g. Qwen2.5-1.5B)|
                                  +--------------------+                          +--------------------+
                                             |                                               |
                                             | Propose K tokens                              | Batch verification
                                             +-----------------------+-----------------------+
                                                                     |
                                                                     v
                                                          +--------------------+
                                                          | Rejection Sampler  |
                                                          +--------------------+
                                                                     |
                                                                     v
                                                          +--------------------+
                                                          | SSE Stream / JSON  |
                                                          +--------------------+
```

## Requirements

- Python 3.10+
- PyTorch 2.2+
- CUDA 12.1+ (recommended for GPU acceleration)

## Installation

```bash
git clone https://github.com/your-org/speculative-inference-proxy.git
cd speculative-inference-proxy

pip install -r requirements.txt
```

## Configuration

Configuration values are read from environment variables or a `.env` file:

| Variable | Type | Default | Description |
|---|---|---|---|
| `TARGET_MODEL_ID` | str | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model ID or local directory for target model |
| `DRAFT_MODEL_ID` | str | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model ID or local directory for draft model |
| `LOOKAHEAD_K` | int | `4` | Number of speculative tokens proposed per cycle |
| `DEVICE` | str | `auto` | Execution device (`cuda`, `cpu`, `mps`, or `auto`) |
| `DTYPE` | str | `auto` | Precision (`bfloat16`, `float16`, `float32`, or `auto`) |
| `PORT` | int | `8000` | HTTP port |
| `HOST` | str | `0.0.0.0` | Bind host |
| `API_KEY` | str | `None` | Optional API key for bearer authentication |
| `USE_MOCK_ENGINE` | bool | `false` | Run with mock token generator (useful for testing/CI) |

## Running the Service

### Local Development

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker compose up -d --build
```

## API Reference

### Chat Completions

`POST /v1/chat/completions`

Accepts standard OpenAI chat completion payloads. Supports both streaming (`stream: true`) and non-streaming responses.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Write a Python function to check for prime numbers."}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": true
  }'
```

### Raw Completions

`POST /v1/completions`

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def fibonacci(n):",
    "max_tokens": 128
  }'
```

### Health & Telemetry

- `GET /health` - Service status and model metadata
- `GET /metrics` - Prometheus metrics (TTFT, ITL, acceptance rate, GPU memory)
- `GET /v1/models` - Active models list

## Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none"
)

# Streaming example
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Explain raft consensus in brief."}],
    max_tokens=200,
    stream=True
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Benchmarks

Run the A/B benchmarking suite against a running instance:

```bash
python benchmarks/benchmark_runner.py --url http://localhost:8000 --tokens 256 --iterations 5
```

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance data and evaluation methodology.
