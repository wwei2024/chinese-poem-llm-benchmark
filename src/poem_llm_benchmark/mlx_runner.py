from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class GenerationResult:
    text: str
    latency_sec: float
    prompt_tokens: int | None
    output_tokens: int | None
    tokens_per_sec: float | None


class MLXModelRunner:
    """Small wrapper around mlx-lm. Load one model at a time on 32 GB Apple Silicon."""

    def __init__(self, model_repo: str):
        try:
            from mlx_lm import load
        except Exception as exc:
            raise RuntimeError("Install mlx-lm first: pip install mlx-lm") from exc
        self.model_repo = model_repo
        self.model, self.tokenizer = load(model_repo)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2, top_p: float = 0.9) -> GenerationResult:
        from mlx_lm import generate
        start = time.perf_counter()
        text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            verbose=False,
        )
        latency = time.perf_counter() - start
        prompt_tokens = _safe_count_tokens(self.tokenizer, prompt)
        output_tokens = _safe_count_tokens(self.tokenizer, text)
        tps = output_tokens / latency if output_tokens is not None and latency > 0 else None
        return GenerationResult(text=text.strip(), latency_sec=latency, prompt_tokens=prompt_tokens, output_tokens=output_tokens, tokens_per_sec=tps)


def _safe_count_tokens(tokenizer, text: str) -> int | None:
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return None
