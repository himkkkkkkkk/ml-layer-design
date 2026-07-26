"""
Layer output extraction for Qwen3-family models.

Core API
--------
    from layer_analysis import LayerAnalyzer

    analyzer = LayerAnalyzer("Qwen/Qwen3-4B-Base")
    states = analyzer.get_hidden_states("hello world", pool="last_token")
    # → torch.Tensor  shape (num_layers+1, hidden_dim)
    #   states[0] = embedding, states[1..N] = decoder layers

    results = analyzer.run_all("hello", pool="mean_content")
    # → dict with hidden_states stats, generated answer, config
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════

def _sep(title: str) -> None:
    print(f"\n{'─'*64}\n  {title}\n{'─'*64}")


# ══════════════════════════════════════════════════════════════════════
# tokenization with content span
# ══════════════════════════════════════════════════════════════════════

def tokenize_with_content(tokenizer, text: str, *, chat: bool = False) -> dict:
    """
    Tokenize and locate the "content" span (excludes chat template tokens).

    Returns dict with keys: input_ids, attention_mask, content_start, content_end.
    """
    if chat:
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full = tokenizer(rendered, return_tensors="pt")
        content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        full_ids = full["input_ids"][0].tolist()
        cs, ce = _find_subsequence(full_ids, content_ids)
    else:
        full = tokenizer(text, return_tensors="pt")
        cs, ce = 0, full["input_ids"].shape[1]

    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "content_start": cs,
        "content_end": ce,
    }


def _find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int]:
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i, i + n
    return 0, len(haystack)


# ══════════════════════════════════════════════════════════════════════
# core: extract hidden states
# ══════════════════════════════════════════════════════════════════════

def extract_hidden_states(model, inputs: dict, *, pool: str = "mean_content") -> torch.Tensor:
    """
    Return stacked hidden states for all layers, pooled over tokens.

    Parameters
    ----------
    pool : {"last_token", "mean_content", "mean_all"}

    Returns
    -------
    torch.Tensor  shape (num_layers + 1, hidden_dim)
        states[0] = embedding, states[1..N] = decoder layers
    """
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )

    if pool == "last_token":
        return torch.stack([h[0, -1, :].float() for h in outputs.hidden_states])
    elif pool == "mean_content":
        cs, ce = inputs["content_start"], inputs["content_end"]
        return torch.stack([h[0, cs:ce, :].float().mean(dim=0) for h in outputs.hidden_states])
    else:  # mean_all
        return torch.stack([h[0, :, :].float().mean(dim=0) for h in outputs.hidden_states])


# ══════════════════════════════════════════════════════════════════════
# LayerAnalyzer
# ══════════════════════════════════════════════════════════════════════

class LayerAnalyzer:
    """Loads a model and extracts hidden states per layer.

    Usage::

        analyzer = LayerAnalyzer("Qwen/Qwen3-4B-Base")
        states = analyzer.get_hidden_states("hello", pool="last_token")
        results = analyzer.run_all("hello", pool="mean_content")
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Base",
        dtype: torch.dtype = torch.float32,
        device_map: str = "auto",
    ):
        self.model_id = model_id
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"Loading model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device_map
        )
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        self.hidden_size = self.model.config.hidden_size
        print(f"  layers={self.num_layers}  hidden={self.hidden_size}  device={self.model.device}")

    # ── generation ─────────────────────────────────────────────────

    def generate(
        self, text: str, *, chat: bool = False, thinking: bool = True,
        max_new_tokens: int = 50, temperature: float = 0.7, top_p: float = 0.95,
    ) -> str:
        """Generate text continuation. Returns empty string if max_new_tokens <= 0."""
        if max_new_tokens <= 0:
            return ""
        if chat:
            messages = [{"role": "user", "content": text}]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            except (TypeError, ValueError):
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            prompt = text
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = generated_ids[0, model_inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    # ── hidden states ──────────────────────────────────────────────

    def get_hidden_states(self, text: str, *, pool: str = "mean_content",
                          chat: bool = False) -> torch.Tensor:
        """Return stacked hidden states tensor. One-liner for the core use case."""
        inputs = tokenize_with_content(self.tokenizer, text, chat=chat)
        return extract_hidden_states(self.model, inputs, pool=pool)

    # ── full run ───────────────────────────────────────────────────

    def run_all(
        self, text: str, *, pool: str = "mean_content", chat: bool = False,
        thinking: bool = True, max_new_tokens: int = 50, temperature: float = 0.7,
        extract: bool = True,
    ) -> dict:
        """Run extraction + generation, return result dict for save_results()."""
        print(f"\nPrompt: {text!r}  chat={chat}  pool={pool}  temp={temperature}\n")

        inputs = tokenize_with_content(self.tokenizer, text, chat=chat)
        n_tokens = inputs["input_ids"].shape[1]

        # generate answer
        _sep("Generated answer")
        answer = self.generate(text, chat=chat, thinking=thinking,
                               max_new_tokens=max_new_tokens, temperature=temperature)
        print(f"  {text}{answer}")

        result: dict[str, Any] = {
            "config": {
                "model_id": self.model_id, "prompt": text,
                "chat": chat, "thinking": thinking, "temperature": temperature,
                "pool": pool, "num_layers": self.num_layers, "hidden_size": self.hidden_size,
                "n_tokens": n_tokens,
                "content_span": (inputs["content_start"], inputs["content_end"]),
                "generated": answer, "full_text": f"{text}{answer}",
            }
        }

        if extract:
            states = extract_hidden_states(self.model, inputs, pool=pool)
            result["hidden_states"] = self._format_hidden_states(states)

        _sep("Done")
        print("All layer outputs extracted successfully.")
        return result

    def _format_hidden_states(self, states: torch.Tensor) -> dict:
        """Convert stacked tensor to stats dict + stacked tensor."""
        _sep("Hidden states")
        labels = ["embedding"] + [f"layer_{i}" for i in range(self.num_layers)]
        stats: dict[str, Any] = {}
        for i, label in enumerate(labels):
            v = states[i]
            n = v.float().norm()
            print(f"  {label:14s}  ‖v‖={n:.2f}  mean={v.mean():+.4f}  "
                  f"std={v.std():.4f}  first3=[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")
            stats[label] = {
                "l2_norm": round(n.item(), 6),
                "mean": round(v.mean().item(), 6),
                "std": round(v.std().item(), 6),
                "min": round(v.min().item(), 6),
                "max": round(v.max().item(), 6),
            }
        cos = torch.nn.functional.cosine_similarity(
            states[0].unsqueeze(0), states[-1].unsqueeze(0), dim=-1
        )
        print(f"\n  cosine-sim(embedding, layer_{self.num_layers-1}) = {cos.item():.6f}")
        stats["cosine_sim_emb_vs_final"] = round(cos.item(), 6)
        stats["_stacked"] = states.cpu()
        return stats
