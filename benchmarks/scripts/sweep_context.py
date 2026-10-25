#!/usr/bin/env python3
"""Context sweep: acceptance and prefill cost from 8k to 32k.

One llama-server per model profile. For each profile, three tasks at prompt
sizes 8192, 16384 and 32768, with two generation budgets where the task does
not stop on its own. Rows are appended to a JSONL file so a rerun skips work
already recorded.

Cold prefill is the first request of a prompt. The second budget repeats the
same prompt, so cache_n shows how much of the prefill was reused.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:8080"
LLAMA_DIR = Path("/home/vlad/opt/llama-b11685")
SERVER = LLAMA_DIR / "llama-server"
CTX = "40960"
REPO = Path("/home/vlad/models/amd-strix-point-llm-perf")
OUT = REPO / "benchmarks/raw/2026-10-25-context-sweep.jsonl"
LOG_DIR = Path("/tmp/llm-perf")

Q38 = "/home/vlad/models/Qwen3.8-27B/Qwen3.8-27B-Q4_K_M.gguf"
Q38_DFLASH = "/home/vlad/models/Qwen3.8-27B-DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
Q38_DSPARK = "/home/vlad/models/Qwen3.8-27B-DSpark/Qwen3.8-27B-DSpark-Q8_0.gguf"
NEMOTRON = "/home/vlad/models/Nemotron-3.5-Bolt-74B-A9B-16GB/Nemotron-3.5-Bolt-74B-A9B-16GB-REAP50-IQ2_XXS.gguf"
Q36_IQ4 = "/home/vlad/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
Q36_Q4 = "/home/vlad/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
Q36_DFLASH = "/home/vlad/models/Qwen3.6-35B-A3B-DFlash2/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf"

ROPE = (
    "У продавца верёвка 30 метров. Каждый день он продаёт ровно 3 метра, "
    "а каждый вечер остаток укорачивается на 10% из-за усушки. "
    "Хватит ли верёвки на 7 дней продаж? Считай остаток после каждого дня."
)
CODE = (
    "Напиши функцию Python is_balanced(s), которая проверяет круглые, "
    "квадратные и фигурные скобки. В ответе только код функции."
)
COUNT = "Count from 1 to 40, comma separated, nothing else."

# (task, thinking, effort, max_tokens list, instruction)
TASKS = [
    ("rope", True, "medium", (128, 512), ROPE),
    ("code", False, "medium", (128, 512), CODE),
    ("count", False, "medium", (200,), COUNT),
]
CONTEXTS = (8192, 16384, 32768)

CONFIGS = [
    {"name": "qwen36-q4-dflash", "model": Q36_Q4, "draft": Q36_DFLASH, "spec": "draft-dflash", "n_max": "7"},
    {"name": "qwen36-q4", "model": Q36_Q4},
    {"name": "qwen36-iq4-dflash", "model": Q36_IQ4, "draft": Q36_DFLASH, "spec": "draft-dflash", "n_max": "7"},
    {"name": "qwen36-iq4", "model": Q36_IQ4},
    {"name": "qwen38-dflash", "model": Q38, "draft": Q38_DFLASH, "spec": "draft-dflash", "n_max": "7"},
    {"name": "qwen38", "model": Q38},
    {"name": "qwen38-mtp5", "model": Q38, "spec": "draft-mtp", "n_max": "5"},
    {"name": "qwen38-dspark", "model": Q38, "draft": Q38_DSPARK, "spec": "draft-dspark", "n_max": "7"},
    {"name": "nemotron-16gb", "model": NEMOTRON, "reasoning_format": "deepseek"},
]

ACCEPT_RE = re.compile(
    r"draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)"
)
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path: str, payload: dict, timeout: int = 1800) -> dict:
    request = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with OPENER.open(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return {}
    return json.loads(body)


def tokenize(text: str) -> int:
    data = post("/tokenize", {"content": text}, timeout=120)
    return len(data["tokens"])


def templated(content: str) -> str | None:
    try:
        data = post(
            "/apply-template",
            {"messages": [{"role": "user", "content": content}]},
            timeout=60,
        )
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        return None
    if isinstance(data, dict):
        prompt = data.get("prompt")
        if isinstance(prompt, str):
            return prompt
    return None


def paragraph(index: int, salt: str) -> str:
    return (
        f"Блок {index} соль {salt}. "
        f"На полке лежит {index * 17 + 3} синих катушек и {index * 13 + 9} красных. "
        f"Запись {index} не повторяет предыдущую.\n"
    )


def template_overhead() -> int:
    sample = "ping"
    rendered = templated(sample)
    if rendered is None:
        return 0
    return max(tokenize(rendered) - tokenize(sample), 0)


def fit_content(instruction: str, target: int, salt: str, overhead: int) -> str:
    """User message whose raw token count plus the chat template is about `target`."""
    unit = paragraph(0, salt)
    unit_tokens = max(tokenize(unit), 1)
    instruction_tokens = tokenize(instruction)
    goal = max(target - overhead - instruction_tokens - 4, unit_tokens)
    blocks = max(goal // unit_tokens, 1)
    filler = "".join(paragraph(i, salt) for i in range(blocks))
    # One correction if the estimate drifted.
    drift = goal - tokenize(filler)
    if abs(drift) > unit_tokens:
        blocks = max(blocks + drift // unit_tokens, 1)
        filler = "".join(paragraph(i, salt) for i in range(blocks))
    return filler + "\n" + instruction


def vram_gib() -> float | None:
    path = Path("/sys/class/drm/card1/device/mem_info_vram_used")
    if not path.exists():
        return None
    return round(int(path.read_text()) / 1024**3, 2)


def server_cmd(cfg: dict) -> list[str]:
    cmd = [str(SERVER), "-m", cfg["model"]]
    if cfg.get("spec"):
        cmd += ["--spec-type", cfg["spec"], "--spec-draft-n-max", cfg["n_max"]]
    if cfg.get("draft"):
        cmd += ["--spec-draft-model", cfg["draft"], "--spec-draft-ngl", "99"]
    cmd += [
        "-ngl",
        "99",
        "-fa",
        "on",
        "-c",
        CTX,
        "--parallel",
        "1",
        "--threads",
        "16",
        "--jinja",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
    if cfg.get("reasoning_format"):
        cmd += ["--reasoning-format", cfg["reasoning_format"]]
    return cmd


def stop_server() -> None:
    subprocess.run(["fuser", "-k", "8080/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)


def start_server(cfg: dict) -> tuple[subprocess.Popen, Path]:
    stop_server()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"sweep-{cfg['name']}.log"
    log = open(log_path, "w")
    env = os.environ.copy()
    env.update(
        {
            "AMD_VULKAN_ICD": "RADV",
            "RADV_PERFTEST": "nogttspill",
            "GGML_VK_VISIBLE_DEVICES": "0",
            "LD_LIBRARY_PATH": str(LLAMA_DIR),
        }
    )
    proc = subprocess.Popen(
        server_cmd(cfg),
        cwd=LLAMA_DIR,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{cfg['name']} exited {proc.returncode}: {log_path}")
        try:
            with OPENER.open(HOST + "/health", timeout=2) as response:
                if response.status == 200:
                    return proc, log_path
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    os.killpg(proc.pid, signal.SIGKILL)
    raise RuntimeError(f"{cfg['name']} did not become healthy: {log_path}")


def stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        stop_server()
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
    stop_server()


def done_keys() -> set[tuple]:
    if not OUT.exists():
        return set()
    keys = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error"):
            continue
        keys.add((row["config"], row["target_ctx"], row["task"], row["max_tokens"]))
    return keys


def latest_accept(log_path: Path, offset: int) -> tuple[int, dict]:
    text = log_path.read_text(errors="replace")
    found = None
    for match in ACCEPT_RE.finditer(text):
        if match.start() >= offset:
            found = match
    new_offset = len(text)
    if not found:
        return new_offset, {}
    return new_offset, {
        "accept_ratio_log": float(found.group(1)),
        "draft_accepted_log": int(found.group(2)),
        "draft_n_log": int(found.group(3)),
        "mean_accept_len": float(found.group(4)),
    }


def run_one(cfg: dict, log_path: Path, log_offset: int, target: int, task: str, thinking: bool, effort: str, max_tokens: int, content: str) -> tuple[dict, int]:
    payload = {
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "reasoning_effort": effort,
        },
    }
    started = time.perf_counter()
    try:
        body = post("/v1/chat/completions", payload)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:500]
        return {
            "config": cfg["name"],
            "target_ctx": target,
            "task": task,
            "max_tokens": max_tokens,
            "error": f"HTTP {exc.code}: {detail}",
        }, log_offset
    wall = time.perf_counter() - started
    choice = body["choices"][0]
    message = choice["message"]
    timings = body.get("timings") or {}
    usage = body.get("usage") or {}
    draft_n = timings.get("draft_n") or 0
    accepted = timings.get("draft_n_accepted")
    accept_pct = None
    if draft_n and accepted is not None:
        accept_pct = round(100.0 * accepted / draft_n, 1)
    log_offset, accept = latest_accept(log_path, log_offset)
    row = {
        "config": cfg["name"],
        "target_ctx": target,
        "task": task,
        "thinking": thinking,
        "max_tokens": max_tokens,
        "wall_s": round(wall, 2),
        "finish": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cache_n": timings.get("cache_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_tok_s": round(timings.get("prompt_per_second") or 0, 2),
        "gen_tok_s": round(timings.get("predicted_per_second") or 0, 2),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": accepted,
        "accept_pct": accept_pct,
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "content_chars": len(message.get("content") or ""),
        "vram_gib": vram_gib(),
    }
    row.update(accept)
    return row, log_offset


def append(row: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: row[k] for k in ("config", "target_ctx", "task", "max_tokens", "prompt_tokens", "gen_tok_s", "accept_pct", "prompt_tok_s", "error") if k in row}, ensure_ascii=False), flush=True)


def main() -> None:
    finished = done_keys()
    for cfg in CONFIGS:
        needed = [
            (target, task, thinking, effort, max_tokens, instruction)
            for target in CONTEXTS
            for task, thinking, effort, budgets, instruction in TASKS
            for max_tokens in budgets
            if (cfg["name"], target, task, max_tokens) not in finished
        ]
        if not needed:
            print(f"skip {cfg['name']}", flush=True)
            continue
        print(f"start {cfg['name']} ({len(needed)} runs)", flush=True)
        proc = None
        try:
            proc, log_path = start_server(cfg)
        except RuntimeError as exc:
            append({"config": cfg["name"], "error": str(exc), "target_ctx": None, "task": None, "max_tokens": None})
            continue
        log_offset = log_path.stat().st_size
        overhead = template_overhead()
        prompts: dict[tuple[int, str], str] = {}
        try:
            for target, task, thinking, effort, max_tokens, instruction in needed:
                key = (target, task)
                if key not in prompts:
                    salt = f"{cfg['name']}-{target}"
                    prompts[key] = fit_content(instruction, target, salt, overhead)
                row, log_offset = run_one(
                    cfg, log_path, log_offset, target, task, thinking, effort, max_tokens, prompts[key]
                )
                append(row)
                finished.add((cfg["name"], target, task, max_tokens))
        finally:
            stop_proc(proc)


if __name__ == "__main__":
    main()
