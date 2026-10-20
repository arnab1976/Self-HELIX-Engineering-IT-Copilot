"""Model providers per tier.

Every tier is an OpenAI-compatible endpoint in production (vLLM, Ollama `/v1`, TGI, OpenAI/Azure for T3).
When a tier has no URL configured, the MockProvider answers with the deterministic `fallback` text that
the calling service already computed from retrieved evidence. That keeps every flow runnable and
testable on a laptop, and makes it obvious which part of an answer comes from the model.

    generate("T1b", messages, fallback="...")            -> LLMResult(text, model, tier, mocked)
    complete_fim("T1a", prompt, fallback="...", stop=[...])
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from common.config import settings
from common.logging import get_logger
from common.tracing import span

log = get_logger("nexus.llm")


@dataclass
class LLMResult:
    text: str
    model: str
    tier: str
    mocked: bool
    latency_ms: float
    usage: dict | None = None


class MockProvider:
    def __init__(self, tier: str):
        self.tier = tier
        self.model = f"mock/{tier.lower()}"

    def chat(self, messages: list[dict], fallback: str = "", **_) -> str:
        return fallback or "I don't have enough information to answer that from the indexed sources."

    def complete(self, prompt: str, fallback: str = "", **_) -> str:
        return fallback

    def stream(self, messages: list[dict], fallback: str = "", **_) -> Iterator[str]:
        text = self.chat(messages, fallback)
        for i in range(0, len(text), 24):
            yield text[i : i + 24]


class OpenAICompatProvider:
    def __init__(self, tier: str, url: str, model: str, api_key: str = "", timeout: float = 60.0):
        self.tier, self.model = tier, model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=timeout)

    def chat(self, messages: list[dict], fallback: str = "", temperature: float = 0.1, max_tokens: int = 800,
             response_format: dict | None = None, **_) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if response_format:   # vLLM / OpenAI structured output (xgrammar-backed in vLLM)
            body["response_format"] = response_format
        r = self.http.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def complete(self, prompt: str, fallback: str = "", max_tokens: int = 64, stop: list[str] | None = None, **_) -> str:
        r = self.http.post("/completions", json={"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                                                 "temperature": 0.0, "stop": stop or ["\n\n"]})
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def stream(self, messages: list[dict], fallback: str = "", **kw) -> Iterator[str]:
        import json as _json

        with self.http.stream("POST", "/chat/completions",
                              json={"model": self.model, "messages": messages, "stream": True, "temperature": 0.1}) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    delta = _json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
                    if delta:
                        yield delta


_CACHE: dict[str, object] = {}


def get_provider(tier: str):
    if tier not in _CACHE:
        url, model, key = settings().tier_endpoint(tier)
        _CACHE[tier] = OpenAICompatProvider(tier, url, model, key) if url else MockProvider(tier)
    return _CACHE[tier]


def generate(tier: str, messages: list[dict], fallback: str = "", **kw) -> LLMResult:
    p = get_provider(tier)
    t0 = time.perf_counter()
    with span("llm.generate", tier=tier, model=p.model):
        try:
            text = p.chat(messages, fallback=fallback, **kw)
            mocked = isinstance(p, MockProvider)
        except (httpx.HTTPError, KeyError) as exc:      # model down -> safe degradation to the evidence-based answer
            log.warning(f"tier {tier} failed: {exc!r}; using fallback")
            text, mocked = fallback, True
    return LLMResult(text, p.model, tier, mocked, round((time.perf_counter() - t0) * 1000, 1))


def complete_fim(tier: str, prompt: str, fallback: str = "", **kw) -> LLMResult:
    p = get_provider(tier)
    t0 = time.perf_counter()
    try:
        text = p.complete(prompt, fallback=fallback, **kw)
        mocked = isinstance(p, MockProvider)
    except httpx.HTTPError:
        text, mocked = "", True          # completion never blocks typing: return no suggestion
    return LLMResult(text, p.model, tier, mocked, round((time.perf_counter() - t0) * 1000, 1))
