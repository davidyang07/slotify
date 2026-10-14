"""llm.py: malformed replies fall back instead of crashing. No network."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from ad_inserter import llm

CANDIDATES = [
    {"ms": 31_000, "snippet": "That wraps up part one."},
    {"ms": 64_000, "snippet": "Anyway."},
    {"ms": 90_000, "snippet": "Okay, moving on."},
]


@pytest.fixture
def stub_openai(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``openai`` module whose chat completion returns ``content``."""

    def install(content: str) -> None:
        class _Completions:
            def create(self, **_kwargs: object) -> SimpleNamespace:
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
        model="stub",
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
        model="stub",
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
        model="stub",
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
        model="stub",
        product_name="Sparrow Notes",
        product_blurb="A calmer note-taking app",
        ad_style="casual",
        ad_mode="DUO",
    )

    assert [(seg.speaker, seg.text) for seg in result.segments] == [
        ("A", "Heard of Sparrow?"),
        ("B", "Every day."),
    ]
