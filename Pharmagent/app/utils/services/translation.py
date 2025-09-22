from __future__ import annotations

import asyncio
import copy
import math
import os
import re
from dataclasses import dataclass
from threading import Lock
import time
from typing import Any, cast

from collections.abc import Callable

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


from Pharmagent.app.api.schemas.clinical import PatientData
from Pharmagent.app.constants import (  
    LANGUAGE_DETECTION_MODEL,
    TRANSLATION_MODEL,   
    MODELS_PATH    
)
from Pharmagent.app.logger import logger
from Pharmagent.app.api.models.providers import initialize_llm_client  






# ###############################################################################
class TranslationService:
    

    def __init__(
        self,
        model_name: str = TRANSLATION_MODEL,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        max_new_tokens: int = 1024,
        batch_size: int = 4,
        beam_size_initial: int = 3,
        beam_size_retry: int = 6,
        temperature_initial: float = 1.0,
        temperature_retry: float = 0.7,
    ) -> None:
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if torch_dtype is None:
            if torch.cuda.is_available():
                # Prefer bfloat16 when available; else float16.
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                torch_dtype = torch.float32
        self.torch_dtype = torch_dtype

        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self._loaded: bool = False

        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self.beam_size_initial = max(1, beam_size_initial)
        self.beam_size_retry = max(self.beam_size_initial + 1, beam_size_retry)
        self.temperature_initial = max(0.1, temperature_initial)
        self.temperature_retry = max(0.1, temperature_retry)

    # -------------------------------------------------------------------------
    def _ensure_model_loaded(self) -> None:
        if self._loaded:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        loaded = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
        )
        self.model = cast(PreTrainedModel, loaded)   
        self.model.to(self.device) # type: ignore
        self.model.eval()
        self._loaded = True

    # -------------------------------------------------------------------------
    def free_model(self) -> None:
        # Release memory if you want to keep the service resident but idle-light.
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        self.model = None
        self.tokenizer = None
        self._loaded = False
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    def _count_words(self, s: str) -> int:
        return len(re.findall(r"\b\w+\b", s, flags=re.UNICODE))    

    # ---------------------------------------------------------------------------
    def translate_text(
        self,
        text: str,
        min_words: int = 50,
        max_words: int = 5000,
        certainty_threshold: float = 0.55,
        max_attempts: int = 2,
        time_budget_s: float | None = None,
    ) -> dict[str, Any]:
        """
        Returns: {
          'translation': str,
          'certainty': float,  # [0,1]
          'attempts': int,
          'latency_ms': float,
        }
        """
        self._ensure_model_loaded()
        n_words = self._count_words(text)
        if n_words < min_words or n_words > max_words:
            raise ValueError(f"text word count out of bounds ({n_words} not in [{min_words}, {max_words}])")

        start = time.perf_counter()
        attempts = 0
        best: dict[str, Any] | None = None

        for attempt in range(1, max(1, max_attempts) + 1):
            attempts = attempt
            use_retry = attempt > 1
            beams = self.beam_size_retry if use_retry else self.beam_size_initial
            temp = self.temperature_retry if use_retry else self.temperature_initial

            translation, certainty = self._translate_large(text, num_beams=beams, temperature=temp)

            if best is None or certainty > best["certainty"]:
                best = {
                    "translation": translation,
                    "certainty": certainty,
                    "attempts": attempts,
                }

            if certainty >= certainty_threshold:
                break

            if time_budget_s is not None:
                spent = time.perf_counter() - start
                if spent >= time_budget_s:
                    break

        latency_ms = (time.perf_counter() - start) * 1000.0
        assert best is not None
        best["latency_ms"] = latency_ms
        return best

    # ---------------------------------------------------------------------------
    def translate_fragments(
        self,
        texts: list[str],
        min_words: int = 50,
        max_words: int = 5000,
        certainty_threshold: float = 0.55,
        max_attempts: int = 2,
        time_budget_s: float | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        per_item_budget: float | None = None
        if time_budget_s is not None and len(texts) > 0:
            per_item_budget = max(0.0, time_budget_s / len(texts))
        for i, t in enumerate(texts):
            tb = None if per_item_budget is None else per_item_budget
            out.append(
                self.translate_text(
                    t,
                    min_words=min_words,
                    max_words=max_words,
                    certainty_threshold=certainty_threshold,
                    max_attempts=max_attempts,
                    time_budget_s=tb,
                )
            )
        return out

    # ---------------------------------------------------------------------------
    async def translate_payload(
        self,
        payload: PatientData,        
        min_words: int = 10,
        max_words: int = 5000,
        certainty_threshold: float = 0.90,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        items = []
        for text in (payload.anamnesis, payload.drugs, payload.exams):            
            if text is None or not text.strip():
                continue
            res = self.translate_text(
                text,
                min_words=min_words,
                max_words=max_words,
                certainty_threshold=certainty_threshold,
                max_attempts=max_attempts,
            )            
            items.append(
                {"translation": res["translation"], "certainty": res["certainty"], "attempts": res["attempts"]}
            )

        avg_certainty = float(sum(d["certainty"] for d in items) / len(items)) if items else math.nan
        report = {"items": items, "avg_certainty": avg_certainty}
        return report

    # ---------------------------------------------------------------------------
    def _translate_large(
        self,
        text: str,
        num_beams: int,
        temperature: float,
    ) -> tuple[str, float]:
        if self.tokenizer is None or self.model is None:
            return "", 0.0
        
        max_src_len = int(self.tokenizer.model_max_length * 0.85)
        chunks = self._chunk_text_by_tokens(text, max_src_len)

        translations: list[str] = []
        probs: list[tuple[float, int]] = []  # (mean_prob, token_count)

        for batch in self._batched(chunks, self.batch_size):
            batch_out = self._translate_batch(batch, num_beams=num_beams, temperature=temperature)
            for s, (seq_prob, tok_count) in zip(batch_out["texts"], batch_out["stats"], strict=True):
                translations.append(s)
                probs.append((seq_prob, tok_count))

        joined = self._join_chunks(translations)

        # Weighted mean probability across chunks
        total_tokens = sum(tc for _, tc in probs) or 1
        certainty = float(sum(p * tc for p, tc in probs) / total_tokens)
        return joined, certainty

    # ---------------------------------------------------------------------------
    def _translate_batch(
        self, batch_src: list[str], num_beams: int, temperature: float
    ) -> dict[str, Any]:
        if self.tokenizer is None or self.model is None:
            return {"texts": [], "stats": []}
        enc = self.tokenizer(
            batch_src,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                num_beams=num_beams,
                do_sample=temperature != 1.0,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                length_penalty=1.0,
            ) # type: ignore

        texts = self.tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)

        # Confidence: mean of chosen-token probabilities across sequence.
        stats: list[tuple[float, int]] = []
        if gen.scores is not None:
            # Align chosen token ids for generated portion
            # sequences shape: [batch, seq_len]; we need the generated tail
            seqs = gen.sequences
            # Determine how many tokens are newly generated for each sample:
            # len(generated) = len(seqs[i]) - input_length[i]
            input_lengths = enc["input_ids"].ne(self.tokenizer.pad_token_id).sum(dim=1)
            offset_per_sample = input_lengths.tolist()

            # gen.scores is a list[timestep] of logits over vocab for each batch.
            # Collect per-sample chosen token probs.
            for i in range(seqs.size(0)):
                chosen_probs: list[float] = []
                for t, logits in enumerate(gen.scores):
                    # token chosen at this step is seqs[i, offset + t]
                    offset = offset_per_sample[i]
                    idx_in_seq = offset + t
                    if idx_in_seq >= seqs.size(1):
                        break
                    token_id = int(seqs[i, idx_in_seq].item())
                    step_logprobs = logits[i].float().log_softmax(dim=-1)
                    chosen_probs.append(float(step_logprobs[token_id].exp().item()))
                tok_count = len(chosen_probs)
                mean_prob = float(sum(chosen_probs) / tok_count) if tok_count else 0.0
                stats.append((mean_prob, tok_count))
        else:
            # Fallback when scores are not returned
            stats = [(0.5, 1) for _ in texts]

        return {"texts": texts, "stats": stats}

    # ---------------------------------------------------------------------------
    def _chunk_text_by_tokens(self, text: str, max_src_len: int) -> list[str]:
        if self.tokenizer is None:
            return [text]
        token_ids = self.tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= max_src_len: # type: ignore
            return [text]

        # Heuristic sentence segmentation to preserve semantics.
        sentences = re.split(r"(?<=[\.\?\!])\s+", text.strip())
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            s_ids = self.tokenizer(s, return_tensors=None, add_special_tokens=False)["input_ids"]
            s_len = len(s_ids) # type: ignore
            if s_len > max_src_len:
                # Hard split on overly long sentences.
                words = s.split()
                buf: list[str] = []
                for w in words:
                    buf.append(w)
                    ids = self.tokenizer(" ".join(buf), return_tensors=None, add_special_tokens=False)[
                        "input_ids"
                    ]
                    if len(ids) >= max_src_len: # type: ignore
                        chunks.append(" ".join(buf))
                        buf = []
                if buf:
                    chunks.append(" ".join(buf))
                cur = []
                cur_len = 0
                continue

            if cur_len + s_len <= max_src_len:
                cur.append(s)
                cur_len += s_len
            else:
                if cur:
                    chunks.append(" ".join(cur))
                cur = [s]
                cur_len = s_len

        if cur:
            chunks.append(" ".join(cur))

        return chunks

    # ---------------------------------------------------------------------------
    def _batched(self, xs: list[str], size: int) -> list[list[str]]:
        return [xs[i : i + size] for i in range(0, len(xs), size)]

    # ---------------------------------------------------------------------------
    def _join_chunks(self, parts: list[str]) -> str:
        # Simple, preserves spacing and sentence boundaries well.
        out = " ".join(p.strip() for p in parts if p.strip())
        # Normalize excessive spaces introduced by joins.
        out = re.sub(r"\s{2,}", " ", out)
        return out