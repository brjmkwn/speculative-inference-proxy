import sys
import time
import argparse
from openai import OpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI SDK client against inference proxy")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="Base URL of the proxy")
    parser.add_argument("--api-key", default="none", help="API key if authentication is enabled")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Connecting to proxy at {args.base_url}...")

    # 1. Models list
    try:
        models = client.models.list()
        print(f"Available models ({len(models.data)}):")
        for m in models.data:
            print(f"  - {m.id}")
    except Exception as e:
        print(f"Error fetching models: {e}")
        return

    # 2. Non-streaming completion
    print("\nTesting non-streaming request...")
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": "Explain binary search in 2 sentences."}
            ],
            max_tokens=80,
            temperature=0.0,
            extra_body={"use_speculative": True, "speculative_k": 4}
        )
        elapsed = time.perf_counter() - start
        content = response.choices[0].message.content
        print(f"Response ({elapsed:.2f}s):\n{content.strip()}")
        if response.usage:
            print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")
    except Exception as e:
        print(f"Non-streaming request failed: {e}")
        return

    # 3. Streaming completion
    print("\nTesting streaming request...")
    start = time.perf_counter()
    chunks = 0
    first_token_time = None
    try:
        stream = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": "Write a quick Python sort function."}
            ],
            max_tokens=80,
            temperature=0.0,
            stream=True,
            extra_body={"use_speculative": True, "speculative_k": 4}
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                sys.stdout.write(chunk.choices[0].delta.content)
                sys.stdout.flush()
                chunks += 1

        total_time = time.perf_counter() - start
        ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
        print(f"\n\nStreaming finished: {chunks} chunks, TTFT: {ttft_ms:.1f}ms, Total: {total_time:.2f}s")

    except Exception as e:
        print(f"Streaming request failed: {e}")
        return


if __name__ == "__main__":
    main()
