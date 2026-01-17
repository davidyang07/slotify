from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResult:
    promo_text: str
    chosen_index: Optional[int]
    rationale: str
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
        "Select the best insertion point that aligns with topic transitions and sentence "
        "boundaries in the context.\n\n"
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
        data = json.loads(raw_text)
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
