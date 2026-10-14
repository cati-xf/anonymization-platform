"""
Minimal Ollama chat-completion client, deliberately independent of
llm_gold_annotation's own Ollama calls -- small enough that duplicating it is
cheaper than coupling two independently-evolving modules together, and keeps
llm_gold_annotation (annotation-assist, already working) untouched. See
LLM_MODULES_PLAN.md's "Shared infra" section.
"""
from __future__ import annotations

import json

import requests

DEFAULT_HOST = "http://localhost:11434"
# 900s (not the read time of any single document -- this corpus's docs run
# well under 100 tokens each, so a 300s timeout tripping is a sign of GPU
# contention or an Ollama-side stall, not slow generation) -- generous
# headroom so a transient stall on a shared GPU node doesn't kill a
# multi-hundred-document run outright.
DEFAULT_TIMEOUT_S = 86400.0


class OllamaError(RuntimeError):
    pass


def chat_json(
    model: str,
    system: str,
    user: str,
    *,
    schema: dict | None = None,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
    max_retries: int = 4,
    think: bool = False,
    timeout: float = DEFAULT_TIMEOUT_S,
    num_ctx: int = 8192,
    num_predict: int = 4096,
    repeat_penalty: float = 1.5,
):
    """
    One system+user chat turn, JSON output constrained to `schema` when given
    (Ollama's structured-output support), else plain "format": "json".
    Retries on invalid JSON.

    `think=False` by default: reasoning-mode models (e.g. Qwen3) can otherwise
    emit chain-of-thought text instead of clean JSON unless explicitly told
    not to -- see LLM_MODULES_PLAN.md's model-choice discussion for why this
    matters here specifically (every track depends on parsing every response).

    `num_ctx=8192` by default: Ollama silently truncates the prompt to its
    own default context window (2048-4096 tokens depending on server
    version) when not told otherwise, with no error -- text beyond the
    window just never reaches the model. Some corpus documents run to
    ~3-4k tokens once the system prompt and JSON-schema overhead are added,
    so the untouched default was plausibly clipping them.

    `num_predict`/`repeat_penalty` only apply starting on the first retry,
    never on the initial attempt -- see below.

    Retry behavior: without any of this, retries resend the exact same
    request, so a genuinely deterministic failure (e.g. a repetition loop
    under temperature=0 greedy decoding) reproduces itself identically on
    every retry -- observed in practice as 3 identical failures in a row.
    Starting on attempt 2, temperature is nudged up (so a retry has an
    actual chance to land somewhere different), `num_predict` is capped at
    4096 (without an explicit cap Ollama lets a response run unbounded
    except by num_ctx, so a model stuck in a repetition loop -- emitting
    the same short JSON fragment over and over instead of terminating --
    fills the entire remaining context before being cut off mid-string,
    turning a fast failure into a slow, expensive one; observed: a single
    tiny document produced a 213k-character truncated response on the
    first attempt), and `repeat_penalty` is raised to 1.5 (Ollama's own
    default is 1.1) to discourage the loop at the source rather than just
    capping the damage once one starts. Even with this, one document (a
    519-byte Fake-corpus doc, where a legitimate response is a few dozen
    tokens) still produced a 10k+ character truncated response across both
    retries under the original 2-retry/0.2-step/1.3-penalty settings --
    `max_retries` was raised from 2 to 4 and the temperature step from 0.2
    to 0.3 (capped at 0.9 instead of 0.8) to give a stubborn loop more
    distinct attempts to land somewhere else. The first attempt
    intentionally keeps Ollama's own defaults for both `num_predict` and
    `repeat_penalty`, so a document that already succeeds on its first
    attempt is generated under exactly the same settings as before this
    retry logic existed -- only documents that were already failing are
    affected.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        options = {"temperature": temperature, "num_ctx": num_ctx}
        if attempt > 0:
            options["temperature"] = min(temperature + attempt * 0.3, 0.9)
            options["num_predict"] = num_predict
            options["repeat_penalty"] = repeat_penalty
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": think,
            "options": options,
            "format": schema if schema is not None else "json",
        }
        resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    raise OllamaError(f"model '{model}' never returned valid JSON after {max_retries + 1} attempt(s): {last_error}")
