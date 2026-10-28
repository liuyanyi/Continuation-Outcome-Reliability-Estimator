from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DIRECT_ANSWER_INSTRUCTION = "Thinking is enough, now give the answer directly"
DIRECT_ANSWER_PROMPT_STYLE_DEFAULT = "assistant"
DIRECT_ANSWER_PROMPT_STYLES = ("assistant", "assistant_empty_think")


def build_direct_answer_prompt(style: str = DIRECT_ANSWER_PROMPT_STYLE_DEFAULT) -> str:
    if style == "assistant":
        assistant_prefix = "<|im_start|>assistant\n"
    elif style == "assistant_empty_think":
        assistant_prefix = "<|im_start|>assistant\n<think></think>"
    else:
        valid = ", ".join(DIRECT_ANSWER_PROMPT_STYLES)
        raise ValueError(f"Unknown direct answer prompt style: {style!r}. Valid: {valid}")

    return (
        "<|im_end|>\n<|im_start|>user\n"
        f"{DIRECT_ANSWER_INSTRUCTION}<|im_end|>\n"
        f"{assistant_prefix}"
    )


DIRECT_ANSWER_PROMPT = build_direct_answer_prompt()

MAX_STEP_CHARS = 2000

REFLECTION_TRIGGER_SETS: dict[str, tuple[str, ...]] = {
    "self_check": (
        "wait",
        "let me check",
        "hold on",
        "have made a mistake",
        "let me double check",
        "wait a moment",
        "is that correct",
        "let me re-read",
    ),
    "strategy_shift": (
        "alternatively",
        "let me try",
        "think of it as",
        "let me consider",
        "what if we try",
        "let's think from a different angle",
        "an alternative method would be",
        "instead of doing that",
    ),
    "uncertainty": (
        "not sure",
        "looks like",
        "that seems",
        "hmm",
        "perhaps",
        "maybe i",
        "i'm not certain",
        "it seems",
        "i suspect",
        "my guess is",
    ),
    "retrospective": (
        "earlier we saw",
        "from before",
        "so now we have",
        "recall that",
        "let me go back",
        "as we established previously",
        "based on our previous result",
        "remember that we found",
        "the value from step",
    ),
}

DEFAULT_REFLECTION_TRIGGER_VOCAB: tuple[str, ...] = tuple(
    phrase
    for phrases in REFLECTION_TRIGGER_SETS.values()
    for phrase in phrases
)

# The paper only gives examples such as "answer is/should be". We keep the cue
# list slightly broader so the stop detector works on our math-style prompts.
DEFAULT_PROVISIONAL_ANSWER_CUES: tuple[str, ...] = (
    "answer is",
    "answer should be",
    "final answer",
    "the final answer is",
    "so the answer is",
    "thus the answer is",
    "therefore the answer is",
    "we get",
    "we have",
)


@dataclass(slots=True)
class StepRange:
    step_index: int
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int


@dataclass(slots=True)
class StepSignal:
    step_index: int
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    similarity: float
    reflective: bool
    trigger_phrase: str | None
    cue_seen_before: bool
    cue_phrase_in_step: str | None
    should_stop: bool
    text_preview: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "similarity": self.similarity,
            "reflective": self.reflective,
            "trigger_phrase": self.trigger_phrase,
            "cue_seen_before": self.cue_seen_before,
            "cue_phrase_in_step": self.cue_phrase_in_step,
            "should_stop": self.should_stop,
            "text_preview": self.text_preview,
        }


@dataclass(slots=True)
class RefrainDecision:
    threshold: float
    steps: list[StepSignal]
    stop_step_index: int | None
    stop_index: int | None
    stop_char_end: int | None
    stop_similarity: float | None
    stop_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "step_count": len(self.steps),
            "triggered": self.stop_step_index is not None,
            "stop_step_index": self.stop_step_index,
            "stop_index": self.stop_index,
            "stop_char_end": self.stop_char_end,
            "stop_similarity": self.stop_similarity,
            "stop_reason": self.stop_reason,
            "steps": [step.to_dict() for step in self.steps],
        }


def normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def find_first_phrase(text: str, phrases: Sequence[str]) -> str | None:
    normalized = normalize_match_text(text)
    for phrase in phrases:
        if phrase in normalized:
            return phrase
    return None


def split_reasoning_steps(response_text: str) -> list[tuple[str, int, int]]:
    if not response_text:
        return []

    blank_line_pattern = re.compile(r"\n\s*\n")
    steps: list[tuple[str, int, int]] = []
    cursor = 0
    while cursor < len(response_text):
        match = blank_line_pattern.search(response_text, cursor)
        raw_end = match.start() if match is not None else len(response_text)
        chunk = response_text[cursor:raw_end]
        stripped = chunk.strip()
        if stripped:
            lead = len(chunk) - len(chunk.lstrip())
            trail = len(chunk.rstrip())
            start = cursor + lead
            end = cursor + trail
            steps.append((response_text[start:end], start, end))
        cursor = match.end() if match is not None else len(response_text)

    if len(steps) > 1:
        return _cap_step_lengths(response_text, steps, max_chars=MAX_STEP_CHARS)

    line_steps = []
    for line_match in re.finditer(r"[^\n]+", response_text):
        line = line_match.group(0)
        stripped = line.strip()
        if not stripped:
            continue
        start = line_match.start() + (len(line) - len(line.lstrip()))
        end = line_match.end() - (len(line) - len(line.rstrip()))
        line_steps.append((response_text[start:end], start, end))
    if len(line_steps) > 1:
        return _cap_step_lengths(response_text, line_steps, max_chars=MAX_STEP_CHARS)

    sentence_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9<])")
    parts: list[tuple[str, int, int]] = []
    start = 0
    for match in sentence_split.finditer(response_text):
        end = match.start()
        chunk = response_text[start:end].strip()
        if chunk:
            inner = response_text[start:end]
            lead = len(inner) - len(inner.lstrip())
            trail = len(inner.rstrip())
            parts.append((response_text[start + lead : start + trail], start + lead, start + trail))
        start = match.end()
    tail = response_text[start:].strip()
    if tail:
        inner = response_text[start:]
        lead = len(inner) - len(inner.lstrip())
        trail = len(inner.rstrip())
        parts.append((response_text[start + lead : start + trail], start + lead, start + trail))
    if parts:
        return _cap_step_lengths(response_text, parts, max_chars=MAX_STEP_CHARS)

    stripped = response_text.strip()
    if not stripped:
        return []
    lead = len(response_text) - len(response_text.lstrip())
    trail = len(response_text.rstrip())
    return _cap_step_lengths(
        response_text,
        [(response_text[lead:trail], lead, trail)],
        max_chars=MAX_STEP_CHARS,
    )


def _cap_step_lengths(
    response_text: str,
    steps: Sequence[tuple[str, int, int]],
    *,
    max_chars: int,
) -> list[tuple[str, int, int]]:
    capped: list[tuple[str, int, int]] = []
    for text, start, end in steps:
        if len(text) <= max_chars:
            capped.append((text, start, end))
            continue
        capped.extend(
            _split_long_step(
                response_text=response_text,
                start=start,
                end=end,
                max_chars=max_chars,
            )
        )
    return capped


def _split_long_step(
    *,
    response_text: str,
    start: int,
    end: int,
    max_chars: int,
) -> list[tuple[str, int, int]]:
    chunks: list[tuple[str, int, int]] = []
    cursor = start
    while cursor < end:
        next_end = min(cursor + max_chars, end)
        if next_end < end:
            window = response_text[cursor:next_end]
            split_at = max(
                window.rfind("\n\n"),
                window.rfind("\n"),
                window.rfind(". "),
                window.rfind("! "),
                window.rfind("? "),
                window.rfind("; "),
                window.rfind(", "),
                window.rfind(" "),
            )
            if split_at > max_chars // 3:
                next_end = cursor + split_at + 1

        raw_chunk = response_text[cursor:next_end]
        stripped = raw_chunk.strip()
        if stripped:
            lead = len(raw_chunk) - len(raw_chunk.lstrip())
            trail = len(raw_chunk.rstrip())
            chunk_start = cursor + lead
            chunk_end = cursor + trail
            if chunk_end > chunk_start:
                chunks.append((response_text[chunk_start:chunk_end], chunk_start, chunk_end))
        cursor = max(next_end, cursor + 1)
    return chunks


def token_char_offsets(
    tokenizer: Any,
    *,
    response_text: str,
    output_ids: Sequence[int],
) -> list[tuple[int, int]]:
    normalized_ids = [int(token_id) for token_id in output_ids]
    encoded = tokenizer(
        response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    encoded_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if (
        isinstance(encoded_ids, list)
        and isinstance(offsets, list)
        and encoded_ids == normalized_ids
        and len(offsets) == len(normalized_ids)
    ):
        return [(int(start), int(end)) for start, end in offsets]

    fallback_offsets: list[tuple[int, int]] = []
    cursor = 0
    response_len = len(response_text)
    for token_id in normalized_ids:
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not piece:
            fallback_offsets.append((cursor, cursor))
            continue

        start = response_text.find(piece, cursor)
        if start < 0:
            lstripped = piece.lstrip()
            if lstripped:
                start = response_text.find(lstripped, cursor)
                if start >= 0:
                    piece = lstripped
        if start < 0:
            start = cursor
        end = min(response_len, start + len(piece))
        cursor = max(cursor, end)
        fallback_offsets.append((start, end))
    return fallback_offsets


def build_step_ranges(
    *,
    response_text: str,
    output_ids: Sequence[int],
    tokenizer: Any,
) -> list[StepRange]:
    if not response_text:
        return []

    steps = split_reasoning_steps(response_text)
    if not steps:
        return []

    offsets = token_char_offsets(
        tokenizer,
        response_text=response_text,
        output_ids=output_ids,
    )
    step_ranges: list[StepRange] = []
    for step_index, (step_text, char_start, char_end) in enumerate(steps):
        token_start: int | None = None
        token_end_inclusive: int | None = None
        for token_idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end > char_start and token_start is None:
                token_start = token_idx
            if tok_start < char_end:
                token_end_inclusive = token_idx

        if token_start is None or token_end_inclusive is None:
            continue
        if token_end_inclusive < token_start:
            continue

        step_ranges.append(
            StepRange(
                step_index=step_index,
                text=step_text,
                char_start=char_start,
                char_end=char_end,
                token_start=token_start,
                token_end=token_end_inclusive + 1,
            )
        )

    if step_ranges:
        return step_ranges

    return [
        StepRange(
            step_index=0,
            text=response_text,
            char_start=0,
            char_end=len(response_text),
            token_start=0,
            token_end=len(output_ids),
        )
    ]


def analyze_refrain(
    *,
    step_ranges: Sequence[StepRange],
    embeddings: Sequence[Sequence[float]],
    threshold: float,
    reflection_vocab: Sequence[str] = DEFAULT_REFLECTION_TRIGGER_VOCAB,
    answer_cues: Sequence[str] = DEFAULT_PROVISIONAL_ANSWER_CUES,
) -> RefrainDecision:
    if len(step_ranges) != len(embeddings):
        raise ValueError(
            "step_ranges and embeddings length mismatch: "
            f"{len(step_ranges)} != {len(embeddings)}"
        )

    if not step_ranges:
        return RefrainDecision(
            threshold=threshold,
            steps=[],
            stop_step_index=None,
            stop_index=None,
            stop_char_end=None,
            stop_similarity=None,
            stop_reason=None,
        )

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    unit = emb / norms

    signals: list[StepSignal] = []
    cue_seen = False
    stop_step_index: int | None = None
    stop_index: int | None = None
    stop_char_end: int | None = None
    stop_similarity: float | None = None
    stop_reason: str | None = None

    for idx, step in enumerate(step_ranges):
        similarity = 0.0
        if idx > 0:
            similarity = float(np.max(np.matmul(unit[:idx], unit[idx])))

        trigger_phrase = find_first_phrase(step.text, reflection_vocab)
        cue_phrase_in_step = find_first_phrase(step.text, answer_cues)
        should_stop = cue_seen and trigger_phrase is not None and similarity >= threshold

        signals.append(
            StepSignal(
                step_index=step.step_index,
                token_start=step.token_start,
                token_end=step.token_end,
                char_start=step.char_start,
                char_end=step.char_end,
                similarity=similarity,
                reflective=trigger_phrase is not None,
                trigger_phrase=trigger_phrase,
                cue_seen_before=cue_seen,
                cue_phrase_in_step=cue_phrase_in_step,
                should_stop=should_stop,
                text_preview=step.text[:200],
            )
        )

        if cue_phrase_in_step is not None:
            cue_seen = True

        if should_stop and stop_step_index is None:
            stop_step_index = step.step_index
            stop_index = step.token_end
            stop_char_end = step.char_end
            stop_similarity = similarity
            stop_reason = (
                "reflection trigger detected after provisional answer cue with "
                f"cosine similarity {similarity:.4f} >= threshold {threshold:.4f}"
            )

    return RefrainDecision(
        threshold=threshold,
        steps=signals,
        stop_step_index=stop_step_index,
        stop_index=stop_index,
        stop_char_end=stop_char_end,
        stop_similarity=stop_similarity,
        stop_reason=stop_reason,
    )
