#!/usr/bin/env python3
"""One non-streaming chat completion against a local llama-server.

Generation speed is the server's timings.predicted_per_second. With thinking
on, that number includes reasoning tokens. Prints one JSON object per repeat.

Example:
  python3 benchmarks/scripts/bench_chat.py \\
    --prompt-file benchmarks/prompts/rope-ru.txt \\
    --max-tokens 512 --thinking on --effort medium --name rope512
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking", choices=("on", "off"), default="on")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--name", default="run")
    args = parser.parse_args()

    content = args.prompt_file.read_text(encoding="utf-8").strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint = args.host.rstrip("/") + "/v1/chat/completions"

    for index in range(1, args.repeats + 1):
        payload = {
            "messages": [{"role": "user", "content": content}],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": args.thinking == "on",
                "reasoning_effort": args.effort,
            },
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with opener.open(request, timeout=900) as response:
            body = json.load(response)
        wall_s = time.perf_counter() - started

        timings = body.get("timings") or {}
        usage = body.get("usage") or {}
        message = body["choices"][0]["message"]
        draft_n = timings.get("draft_n") or 0
        draft_accepted = timings.get("draft_n_accepted")
        accept_pct = None
        if draft_n and draft_accepted is not None:
            accept_pct = round(100.0 * draft_accepted / draft_n, 1)

        name = args.name if args.repeats == 1 else f"{args.name}-{index}"
        row = {
            "name": name,
            "wall_s": round(wall_s, 2),
            "finish": body["choices"][0].get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tok_s": round(timings.get("prompt_per_second") or 0, 2),
            "gen_tok_s": round(timings.get("predicted_per_second") or 0, 2),
            "predicted_n": timings.get("predicted_n"),
            "predicted_ms": timings.get("predicted_ms"),
            "draft_n": timings.get("draft_n"),
            "draft_n_accepted": draft_accepted,
            "accept_pct": accept_pct,
            "reasoning_chars": len(message.get("reasoning_content") or ""),
            "content_chars": len(message.get("content") or ""),
        }
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
