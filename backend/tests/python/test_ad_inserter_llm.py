"""llm.py: malformed replies fall back, replies and calls are capped. No network."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Dict, List

import pytest

from ad_inserter import llm

CANDIDATES = [
    {"ms": 31_000, "snippet": "That wraps up part one."},
    {"ms": 64_000, "snippet": "Anyway."},
    {"ms": 90_000, "snippet": "Okay, moving on."},
]


MODEL = llm.DEFAULT_OPENAI_MODEL


@pytest.fixture
def openai_calls() -> List[Dict[str, object]]:
    """The keyword arguments of every chat completion the fake client served."""
    return []


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch, openai_calls: List[Dict[str, object]]):
    """Install a fake ``openai`` module whose chat completion returns ``content``."""

    def install(content: str) -> None:
        class _Completions:
            def create(self, **kwargs: object) -> SimpleNamespace:
                openai_calls.append(kwargs)
                message = SimpleNamespace(content=content)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        class OpenAI:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.chat = SimpleNamespace(completions=_Completions())

        module = types.ModuleType("openai")
        module.OpenAI = OpenAI  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "openai", module)

    return install


def _choose() -> llm.LLMResult:
    return llm.generate_promo_and_choice(
        provider="openai",
        model=MODEL,
        product_name="Sparrow Notes",
        product_desc="A calmer note-taking app",
        product_url=None,
        mode="podcast",
        candidates=CANDIDATES,
    )


@pytest.mark.parametrize(
    "reply",
    [
        "this is not json",
        '{"promo_text": "Try Sparrow.", "chosen_index": "2nd", "rationale": "x"}',
        '[1, 2, 3]',
        '"just a string"',
        '{"chosen_index": 1, "rationale": "no promo"}',
        '{"promo_text": "Try Sparrow.", "rationale": "no index"}',
        '{"promo_text": "Try Sparrow.", "chosen_index": 7}',
        '{"promo_text": "Try Sparrow.", "chosen_index": -1}',
        '{"promo_text": "Try Sparrow.", "chosen_index": 1.5}',
        '{"promo_text": "Try Sparrow.", "chosen_index": true}',
        '{"promo_text": "Try Sparrow.", "chosen_index": [1]}',
        '{"promo_text": null, "chosen_index": 1}',
        '{"promo_text": 42, "chosen_index": 1}',
    ],
)
def test_malformed_choice_falls_back_like_invalid_json(stub_openai, reply: str) -> None:
    stub_openai(reply)

    result = _choose()

    assert result.promo_text == "Sparrow Notes"
    assert result.chosen_index is None
    assert "falling back to heuristic insertion" in result.rationale
    assert result.raw_text == reply


@pytest.mark.parametrize(
    ("reply", "expected_index"),
    [
        ('{"promo_text": " Try Sparrow. ", "chosen_index": 2, "rationale": "r"}', 2),
        ('```json\n{"promo_text": "Try Sparrow.", "chosen_index": "1"}\n```', 1),
        ('Sure! {"promo_text": "Try Sparrow.", "chosen_index": 0.0}', 0),
        ('{"promo_text": "Try Sparrow.", "chosen_index": null}', None),
    ],
)
def test_valid_choice_is_used(stub_openai, reply: str, expected_index) -> None:
    stub_openai(reply)

    result = _choose()

    assert result.promo_text == "Try Sparrow."
    assert result.chosen_index == expected_index


def test_no_candidates_means_no_valid_index(monkeypatch, stub_openai) -> None:
    stub_openai('{"promo_text": "Try Sparrow.", "chosen_index": 0}')

    result = llm.generate_promo_and_choice(
        provider="openai",
        model=MODEL,
        product_name="Sparrow Notes",
        product_desc="A calmer note-taking app",
        product_url=None,
        mode="podcast",
        candidates=[],
    )

    assert result.chosen_index is None
    assert result.promo_text == "Sparrow Notes"


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        '["A", "B"]',
        '{"segments": "nope"}',
        '{"segments": [{"speaker": "C", "text": "hi"}, "junk", 3]}',
    ],
)
def test_malformed_ad_script_falls_back(stub_openai, reply: str) -> None:
    stub_openai(reply)

    result = llm.generate_ad_script(
        provider="openai",
        model=MODEL,
        product_name="Sparrow Notes",
        product_blurb="A calmer note-taking app",
        ad_style="casual",
        ad_mode="DUO",
    )

    assert [seg.speaker for seg in result.segments] == ["A", "B"]
    assert result.segments[1].text == "A calmer note-taking app"


def test_valid_ad_script_is_parsed(stub_openai) -> None:
    stub_openai(
        '{"segments": [{"speaker": "a", "text": "Heard of Sparrow?"},'
        ' {"speaker": "B", "text": "Every day."}]}'
    )

    result = llm.generate_ad_script(
        provider="openai",
        model=MODEL,
        product_name="Sparrow Notes",
        product_blurb="A calmer note-taking app",
        ad_style="casual",
        ad_mode="DUO",
    )

    assert [(seg.speaker, seg.text) for seg in result.segments] == [
        ("A", "Heard of Sparrow?"),
        ("B", "Every day."),
    ]


def _script(**overrides: object) -> llm.AdScriptResult:
    kwargs: Dict[str, object] = dict(
        provider="openai",
        model=MODEL,
        product_name="Sparrow Notes",
        product_blurb="A calmer note-taking app",
        ad_style="casual",
        ad_mode="DUO",
    )
    kwargs.update(overrides)
    return llm.generate_ad_script(**kwargs)  # type: ignore[arg-type]


def test_every_openai_call_has_a_completion_budget(stub_openai, openai_calls) -> None:
    stub_openai('{"promo_text": "Try Sparrow.", "chosen_index": 0}')

    _choose()
    _script()

    assert len(openai_calls) == 2
    for call in openai_calls:
        assert call["model"] == MODEL
        assert call["max_tokens"] == llm.OPENAI_MAX_TOKENS
    assert 100 <= llm.OPENAI_MAX_TOKENS <= 1000  # a few hundred: a ~20 s read


@pytest.mark.parametrize("generate", ["choice", "script"])
def test_a_model_off_the_allow_list_is_refused_before_any_request(
    stub_openai, openai_calls, generate: str
) -> None:
    stub_openai('{"promo_text": "Try Sparrow.", "chosen_index": 0}')

    with pytest.raises(llm.ModelNotAllowed, match="'gpt-4o' is not allowed.*gpt-4o-mini"):
        if generate == "choice":
            llm.generate_promo_and_choice(
                provider="openai",
                model="gpt-4o",
                product_name="Sparrow Notes",
                product_desc="A calmer note-taking app",
                product_url=None,
                mode="podcast",
                candidates=CANDIDATES,
            )
        else:
            _script(model="gpt-4o")

    assert openai_calls == []


def test_the_allow_list_comes_from_slotify_openai_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert llm.allowed_openai_models() == ["gpt-4o-mini"]
    assert llm.default_openai_model() == "gpt-4o-mini"

    monkeypatch.setenv("SLOTIFY_OPENAI_MODELS", " gpt-4.1-mini , ,gpt-4o ")
    assert llm.allowed_openai_models() == ["gpt-4.1-mini", "gpt-4o"]
    assert llm.default_openai_model() == "gpt-4.1-mini"
    assert llm.check_openai_model("gpt-4o") == "gpt-4o"
    with pytest.raises(llm.ModelNotAllowed):
        llm.check_openai_model("gpt-4o-mini")

    monkeypatch.setenv("SLOTIFY_OPENAI_MODELS", "gpt-4o,gpt-4o-mini")
    assert llm.default_openai_model() == "gpt-4o-mini"

    monkeypatch.setenv("SLOTIFY_OPENAI_MODELS", " , ")
    assert llm.allowed_openai_models() == ["gpt-4o-mini"]


def test_resolve_model_checks_only_the_openai_provider() -> None:
    assert llm.resolve_model("openai", None) == "gpt-4o-mini"
    assert llm.resolve_model("none", "anything") == "anything"
    with pytest.raises(llm.ModelNotAllowed):
        llm.resolve_model("openai", "gpt-4")


def test_the_script_is_capped_at_max_segments(stub_openai) -> None:
    lines = [{"speaker": "AB"[i % 2], "text": f"Line {i}."} for i in range(40)]
    stub_openai(json.dumps({"segments": lines}))

    result = _script()

    assert llm.MAX_AD_SEGMENTS == 6
    assert [seg.text for seg in result.segments] == [f"Line {i}." for i in range(6)]


def test_overlong_script_text_is_cut_on_a_word_boundary(stub_openai) -> None:
    rambling = " ".join(["sparrow"] * 300)
    stub_openai(
        json.dumps(
            {
                "segments": [
                    {"speaker": "A", "text": "Heard of Sparrow?"},
                    {"speaker": "B", "text": rambling},
                    {"speaker": "A", "text": "Nope."},
                ]
            }
        )
    )

    result = _script()

    assert [seg.speaker for seg in result.segments] == ["A", "B"]
    assert sum(len(seg.text) for seg in result.segments) <= llm.MAX_SCRIPT_CHARS
    assert set(result.segments[1].text.split()) == {"sparrow"}


@pytest.mark.parametrize("ad_mode", ["A_ONLY", "B_ONLY", "DUO"])
def test_fallback_scripts_cap_the_product_text(ad_mode: str) -> None:
    result = _script(provider="none", product_blurb="word " * 1000, ad_mode=ad_mode)

    assert result.segments
    assert sum(len(seg.text) for seg in result.segments) <= llm.MAX_SCRIPT_CHARS


def test_promo_text_is_capped(stub_openai) -> None:
    stub_openai(json.dumps({"promo_text": "Try Sparrow today. " * 100, "chosen_index": 1}))

    result = _choose()

    assert result.chosen_index == 1
    assert 0 < len(result.promo_text) <= llm.MAX_SCRIPT_CHARS
    assert result.promo_text.endswith("today.")


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        ("  short  ", 10, "short"),
        ("hello world foo", 11, "hello world"),
        ("hello world foo", 13, "hello world"),
        ("hello world foo", 8, "hello"),
        ("unbreakable", 6, "unbrea"),
        ("anything", 0, ""),
    ],
)
def test_clip_text(text: str, limit: int, expected: str) -> None:
    assert llm.clip_text(text, limit) == expected
