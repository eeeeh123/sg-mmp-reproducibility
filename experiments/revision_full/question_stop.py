"""Auditable generated-only stopping at a repeated few-shot question marker.

The historical evaluator already ignores text after the next ``Question:``
marker when extracting an answer.  This module makes that rule an online,
per-sequence stopping policy without inspecting the prompt or changing the
answer extractor.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


STOP_PROTOCOL = "generated-question-marker-v1"
QUESTION_MARKER_RE = re.compile(r"(?:^|\n[ \t]*\n)[ \t]*Question[ \t]*:")
BASE_GENERATION_KWARGS = {
    "do_sample": False,
    "max_new_tokens": 256,
    "padding_side": "left",
}


def json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


BASE_GENERATION_KWARGS_SHA256 = json_sha256(BASE_GENERATION_KWARGS)


@dataclass(frozen=True)
class CanonicalGeneration:
    text: str
    marker_found: bool
    marker_start: int | None


def canonical_answer_prefix(text: str) -> CanonicalGeneration:
    """Return exactly the generated answer prefix before a new question.

    A marker is recognized only at the beginning of generated text or after a
    blank generated line.  Therefore ``Question:`` in the immutable few-shot
    prompt, or in an ordinary reasoning sentence, cannot trigger this rule.
    """

    match = QUESTION_MARKER_RE.search(text)
    if match is None:
        return CanonicalGeneration(text=text, marker_found=False, marker_start=None)
    prefix = text[: match.start()]
    return CanonicalGeneration(
        text=prefix.rstrip(), marker_found=True, marker_start=match.start()
    )


class GeneratedQuestionStopLogitsProcessor:
    """Force EOS independently for rows that generated a new question marker."""

    def __init__(self, tokenizer, prompt_width: int, eos_token_id: int):
        if prompt_width < 0:
            raise ValueError("prompt_width must be non-negative")
        if eos_token_id is None:
            raise ValueError("eos_token_id is required")
        self.tokenizer = tokenizer
        self.prompt_width = int(prompt_width)
        self.eos_token_id = int(eos_token_id)
        self.finished: list[bool] | None = None

    def __call__(self, input_ids, scores):
        import torch

        batch = int(input_ids.shape[0])
        if self.finished is None:
            self.finished = [False] * batch
        if len(self.finished) != batch:
            raise RuntimeError("generation batch size changed during decoding")

        generated = input_ids[:, self.prompt_width :]
        for row in range(batch):
            if not self.finished[row]:
                text = self.tokenizer.decode(
                    generated[row], skip_special_tokens=True
                )
                self.finished[row] = canonical_answer_prefix(text).marker_found
            if self.finished[row]:
                scores[row].fill_(torch.finfo(scores.dtype).min)
                scores[row, self.eos_token_id] = 0
        return scores


def generation_diagnostics(
    raw_text: str,
    token_ids: list[int],
    *,
    eos_token_id: int,
    max_new_tokens: int,
) -> dict:
    canonical = canonical_answer_prefix(raw_text)
    eos_positions = [
        index for index, token_id in enumerate(token_ids) if token_id == eos_token_id
    ]
    ended_with_eos = bool(eos_positions)
    generated_token_count = eos_positions[0] + 1 if ended_with_eos else len(token_ids)
    if canonical.marker_found:
        stop_reason = "generated_question_marker"
    elif ended_with_eos:
        stop_reason = "model_eos"
    elif generated_token_count >= max_new_tokens:
        stop_reason = "max_new_tokens"
    else:
        stop_reason = "unknown"
    return {
        "generation": canonical.text,
        "raw_generation": raw_text,
        "generated_question_marker_found": canonical.marker_found,
        "generated_question_marker_start": canonical.marker_start,
        "generated_token_count": generated_token_count,
        "ended_with_eos": ended_with_eos,
        "truncated": stop_reason == "max_new_tokens",
        "stop_reason": stop_reason,
        "stop_protocol": STOP_PROTOCOL,
        "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
    }
