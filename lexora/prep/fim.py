"""Fill-in-the-middle sample construction for L4 (spec 4.4).

Each document is split at two random points into prefix / middle / suffix and rewritten in PSM or SPM
order with the base model's special tokens. Loss is applied to the middle only. A `fim_rate` share of
documents is transformed; the rest stay left-to-right so the model keeps both skills.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

FIM_TOKENS = {
    # model family -> (prefix, suffix, middle, eos)
    "qwen": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>"),
    "starcoder": ("<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<|endoftext|>"),
    "codellama": ("<PRE> ", " <SUF>", " <MID>", "</s>"),
    "deepseek": ("<｜fim▁begin｜>", "<｜fim▁hole｜>", "<｜fim▁end｜>", "<｜end▁of▁sentence｜>"),
}


@dataclass
class FIMSample:
    text: str
    mode: str          # psm | spm | ar
    prefix: str = ""
    middle: str = ""
    suffix: str = ""


def make_sample(doc: str, rng: random.Random, family: str = "qwen", fim_rate: float = 0.5, spm_rate: float = 0.5) -> FIMSample:
    P, S, M, E = FIM_TOKENS[family]
    if len(doc) < 20 or rng.random() >= fim_rate:
        return FIMSample(doc + E, "ar")
    a, b = sorted(rng.sample(range(1, len(doc)), 2))
    prefix, middle, suffix = doc[:a], doc[a:b], doc[b:]
    if rng.random() < spm_rate:
        return FIMSample(f"{P}{S}{suffix}{M}{prefix}{middle}{E}", "spm", prefix, middle, suffix)
    return FIMSample(f"{P}{prefix}{S}{suffix}{M}{middle}{E}", "psm", prefix, middle, suffix)


def build_prompt(prefix: str, suffix: str, family: str = "qwen", context: str = "") -> str:
    """Inference-time PSM prompt; repo context is prepended as comments, outside the FIM span."""
    P, S, M, _ = FIM_TOKENS[family]
    return f"{context}{P}{prefix}{S}{suffix}{M}"
