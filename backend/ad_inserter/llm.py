from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResult:
    promo_text: str
    chosen_index: Optional[int]
    rationale: str
    prompt: str
    raw_text: str


@dataclass
class AdSegment:
    speaker: str
    text: str


@dataclass
class AdScriptResult:
    segments: List[AdSegment]
    prompt: str
    raw_text: str


def _build_prompt(
    product_name: str,
    product_desc: str,
    product_url: Optional[str],
    mode: str,
    candidates: List[Dict[str, Any]],
) -> str:
    url_line = f"\nProduct URL: {product_url}" if product_url else ""
    candidate_lines = []
    for idx, cand in enumerate(candidates):
        snippet = cand.get("snippet", "")
        candidate_lines.append(
            f"[{idx}] t={cand['ms']}ms\nContext before: {snippet}"
        )
    candidate_block = "\n\n".join(candidate_lines) if candidate_lines else "No candidates."

    return (
        "You are writing a native-sounding, 1-sentence promo that should match the tone "
        "of the preceding audio. It should be 8-12 seconds when spoken aloud.\n\n"
        f"Mode: {mode}\n"
        f"Product name: {product_name}\n"
        f"Product description: {product_desc}{url_line}\n\n"
        "Identify insertion points in the following text. An insertion point is valid ONLY if it meets ALL of these criteria:"

"It must be at the end of a complete sentence (after punctuation like . ! ?)"
"It must be followed by a natural pause in the narrative or dialogue"
"It should NOT be in the middle of:"

"A continuous action sequence"
"Back-and-forth dialogue between characters"
"A single character's uninterrupted speech"
"A flowing description of a single scene or moment"
        "Never choose the start or the end of the audio clip unless absolutely necessary "
        "(e.g., the clip is just one sentence).\n\n"
        "Candidates:\n"
        f"{candidate_block}\n\n"
        "Respond ONLY in JSON with keys: promo_text, chosen_index, rationale."
    )


def generate_promo_and_choice(
    provider: str,
    model: str,
    product_name: str,
    product_desc: str,
    product_url: Optional[str],
    mode: str,
    candidates: List[Dict[str, Any]],
) -> LLMResult:
    prompt = _build_prompt(
        product_name=product_name,
        product_desc=product_desc,
        product_url=product_url,
        mode=mode,
        candidates=candidates,
    )

    if provider == "none":
        return LLMResult(
            promo_text=product_name,
            chosen_index=None,
            rationale="LLM disabled; using heuristic insertion and product name only.",
            prompt=prompt,
            raw_text="",
        )

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful audio editor and ad writer.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        raw_text = response.choices[0].message.content or ""
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    promo_text = product_name
    chosen_index: Optional[int] = None
    rationale = ""

    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)
        if not cleaned.startswith("{"):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(0)
        data = json.loads(cleaned)
        promo_text = str(data.get("promo_text", promo_text)).strip()
        chosen_index_val = data.get("chosen_index")
        if chosen_index_val is not None:
            chosen_index = int(chosen_index_val)
        rationale = str(data.get("rationale", "")).strip()
    except json.JSONDecodeError:
        rationale = "LLM response was not valid JSON; falling back to heuristic insertion."

    return LLMResult(
        promo_text=promo_text,
        chosen_index=chosen_index,
        rationale=rationale,
        prompt=prompt,
        raw_text=raw_text,
    )


def _build_ad_script_prompt(
    product_name: str,
    product_blurb: str,
    ad_style: str,
    ad_mode: str,
) -> str:
    return (
        "You are writing a short, natural-sounding ad read for a two-person conversation. "
        "Keep it concise (8-15 seconds total)."
        "\n\n"
        f"Product name: {product_name}\n"
        f"Product blurb: {product_blurb}\n"
        f"Ad style: {ad_style}\n"
        f"Ad mode: {ad_mode}\n\n"
        "Return ONLY machine-readable JSON in this exact shape:\n"
        '{ "segments": [ { "speaker": "A", "text": "..." }, '
        '{ "speaker": "B", "text": "..." } ] }\n'
        "Rules:\n"
        "- speaker must be A or B only.\n"
        "- If ad_mode is A_ONLY or B_ONLY, include only that speaker.\n"
        "- If ad_mode is DUO, use a short back-and-forth (2-4 segments).\n"
        "- Avoid emojis and hashtags.\n"
    )


def _parse_ad_segments(raw_text: str) -> List[AdSegment]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
    data = json.loads(cleaned)
    segments_raw = data.get("segments", [])
    segments: List[AdSegment] = []
    if isinstance(segments_raw, list):
        for item in segments_raw:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("speaker", "")).strip().upper()
            text = str(item.get("text", "")).strip()
            if speaker in {"A", "B"} and text:
                segments.append(AdSegment(speaker=speaker, text=text))
    return segments


def generate_ad_script(
    provider: str,
    model: str,
    product_name: str,
    product_blurb: str,
    ad_style: str,
    ad_mode: str,
) -> AdScriptResult:
    """Generate a structured multi-speaker ad script."""
    prompt = _build_ad_script_prompt(
        product_name=product_name,
        product_blurb=product_blurb,
        ad_style=ad_style,
        ad_mode=ad_mode,
    )

    if provider == "none":
        if ad_mode == "B_ONLY":
            segments = [AdSegment(speaker="B", text=f"{product_name} {product_blurb}")]
        elif ad_mode == "DUO":
            segments = [
                AdSegment(speaker="A", text=f"Have you tried {product_name}?"),
                AdSegment(speaker="B", text=product_blurb),
            ]
        else:
            segments = [AdSegment(speaker="A", text=f"{product_name} {product_blurb}")]
        return AdScriptResult(segments=segments, prompt=prompt, raw_text="")

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You write short, structured ad scripts.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        raw_text = response.choices[0].message.content or ""
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    try:
        segments = _parse_ad_segments(raw_text)
    except json.JSONDecodeError:
        segments = []

    if not segments:
        if ad_mode == "B_ONLY":
            segments = [AdSegment(speaker="B", text=f"{product_name} {product_blurb}")]
        elif ad_mode == "DUO":
            segments = [
                AdSegment(speaker="A", text=f"Quick shout-out to {product_name}."),
                AdSegment(speaker="B", text=product_blurb),
            ]
        else:
            segments = [AdSegment(speaker="A", text=f"{product_name} {product_blurb}")]

    return AdScriptResult(segments=segments, prompt=prompt, raw_text=raw_text)
