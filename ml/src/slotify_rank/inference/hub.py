"""Pinned Hugging Face snapshots: which ones serving needs, whether they are here.

Serving loads three models -- Whisper (transcription), the Whisper encoder (audio
embeddings) and MiniLM (text embeddings) -- through ``from_pretrained``, at the
``model_revision`` the committed ``ml/configs/*.yaml`` name. Two things follow
from a revision of ``main``:

* **Serving can silently diverge from training.** ``main`` is whatever the Hub
  serves today; the day upstream moves it, a fresh cache serves different
  weights while every config digest and cache identity still says ``main``.
* **Every load talks to the Hub.** Even with the weights cached,
  ``transformers`` resolves ``main`` and probes optional files over the network,
  so a slow or unreachable Hub slows or breaks ranking.

Pinning the revision to a commit SHA fixes the first. The second is fixed by
running offline (``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1``) whenever
every pinned snapshot is already in the local cache -- which is what this module
decides. ``infer warm`` is how the snapshots get there.

The cache lookup mirrors where the loaders themselves look: ``transformers``
honours ``TRANSFORMERS_CACHE`` and then the Hugging Face hub cache
(``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE`` / ``HF_HOME`` / ``XDG_CACHE_HOME``);
``sentence-transformers`` honours ``SENTENCE_TRANSFORMERS_HOME`` and then the
same hub cache. Inside it, a snapshot is ``models--<org>--<name>/snapshots/<sha>/``.
Nothing here imports ``huggingface_hub``: those libraries read the offline flag
once, at import, so the decision has to be made before any of them is loaded.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

__all__ = [
    "COMMIT_SHA",
    "HubPlan",
    "OFFLINE_ENV",
    "PinnedModel",
    "SnapshotsMissing",
    "WARM_COMMAND",
    "apply_offline_env",
    "hub_cache_dir",
    "pinned_models",
    "plan_hub_access",
    "snapshot_dir",
    "snapshot_present",
    "warm_models",
]

#: A full Hugging Face commit SHA. Anything else (``main``, a tag, a short SHA)
#: names something that can move.
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

#: The command that fetches the pinned snapshots. Named in every error that is
#: fixed by running it.
WARM_COMMAND = "python -m slotify_rank.cli infer warm"

#: What offline mode sets. Both, because ``huggingface_hub`` reads the first and
#: older ``transformers`` code paths read the second.
OFFLINE_ENV: tuple[str, ...] = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

#: The libraries that read :data:`OFFLINE_ENV` once, when first imported.
_HUB_LIBRARIES: tuple[str, ...] = (
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
)

#: A snapshot is usable when it holds a config and the weights. The loaders
#: fetch more (tokenizer, preprocessor, pooling config); a snapshot missing those
#: still fails loudly at load, offline, naming :data:`WARM_COMMAND`.
_WEIGHT_FILES: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin")


class SnapshotsMissing(RuntimeError):
    """A pinned snapshot serving needs is not in the local cache."""


@dataclass(frozen=True)
class PinnedModel:
    """One model at one revision, and which configs ask for it."""

    model_id: str
    revision: str
    #: ``transformers`` or ``sentence-transformers``: which loader, so which cache.
    library: str
    #: The config sections that load it, e.g. ``("transcription", "embeddings.audio")``.
    used_by: tuple[str, ...]

    @property
    def is_pinned(self) -> bool:
        return bool(COMMIT_SHA.match(self.revision))

    def describe(self) -> str:
        return f"{self.model_id}@{self.revision}"


def pinned_models(configs: Any) -> tuple[PinnedModel, ...]:
    """The distinct (model, revision, loader) triples a set of pipeline configs loads.

    Whisper appears once when transcription and the audio encoder name the same
    revision -- which they must, for the transcript and the speech
    representation to come from the same weights -- and twice if they disagree,
    so the disagreement is visible rather than collapsed.
    """
    wanted = (
        ("transcription", configs.transcription, "transformers"),
        ("embeddings.audio", configs.embeddings.audio, "transformers"),
        ("embeddings.text", configs.embeddings.text, "sentence-transformers"),
    )
    merged: dict[tuple[str, str, str], list[str]] = {}
    for section, config, library in wanted:
        key = (str(config.model_id), str(config.model_revision), library)
        merged.setdefault(key, []).append(section)
    return tuple(
        PinnedModel(model_id=model_id, revision=revision, library=library, used_by=tuple(sections))
        for (model_id, revision, library), sections in merged.items()
    )


def hub_cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The Hugging Face hub cache, resolved exactly as ``huggingface_hub`` does."""
    env = os.environ if environ is None else environ

    def expand(value: str) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(value)))

    explicit = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return expand(explicit)
    home = env.get("HF_HOME")
    if home:
        return expand(home) / "hub"
    xdg = env.get("XDG_CACHE_HOME") or os.path.join("~", ".cache")
    return expand(xdg) / "huggingface" / "hub"


def _cache_root(model: PinnedModel, environ: Mapping[str, str]) -> Path:
    override = (
        environ.get("SENTENCE_TRANSFORMERS_HOME")
        if model.library == "sentence-transformers"
        else environ.get("TRANSFORMERS_CACHE")
    )
    if override:
        return Path(os.path.expandvars(os.path.expanduser(override)))
    return hub_cache_dir(environ)


def snapshot_dir(model: PinnedModel, environ: Mapping[str, str] | None = None) -> Path:
    """Where ``model``'s snapshot lives (or would live) in the local cache."""
    env = os.environ if environ is None else environ
    folder = "models--" + model.model_id.replace("/", "--")
    return _cache_root(model, env) / folder / "snapshots" / model.revision


def snapshot_present(model: PinnedModel, environ: Mapping[str, str] | None = None) -> bool:
    """True when the pinned snapshot holds a config and weights.

    Only a pinned revision can be present: the directory for ``main`` would be a
    SHA, and which SHA ``main`` meant is exactly what is not recorded.
    """
    if not model.is_pinned:
        return False
    directory = snapshot_dir(model, environ)
    return (directory / "config.json").is_file() and any(
        (directory / name).is_file() for name in _WEIGHT_FILES
    )


@dataclass(frozen=True)
class HubPlan:
    """What the local cache allows: offline serving, or not, and why."""

    models: tuple[PinnedModel, ...]
    #: Pinned, but the snapshot is not in the local cache.
    missing: tuple[PinnedModel, ...]
    #: Not pinned to a commit SHA, so offline mode cannot know what to load.
    unpinned: tuple[PinnedModel, ...]

    @property
    def offline(self) -> bool:
        """Every model pinned and present: nothing to fetch, nothing to resolve."""
        return bool(self.models) and not self.missing and not self.unpinned

    def missing_message(self) -> str:
        # The command comes first: a client sees only the first ~300 characters
        # of an error line, and the fix matters more than the SHAs.
        names = ", ".join(model.describe() for model in self.missing)
        return (
            "Pinned model snapshot(s) are not in the local Hugging Face cache; run "
            f"`{WARM_COMMAND}` (from ml/, with network access) to fetch exactly "
            f"those revisions: {names}."
        )

    def unpinned_message(self) -> str:
        names = ", ".join(
            f"{model.model_id} ({'/'.join(model.used_by)}: {model.revision!r})"
            for model in self.unpinned
        )
        return (
            f"Model revision(s) not pinned to a commit SHA: {names}. They are "
            "resolved against the Hugging Face Hub on every load, so serving needs "
            "the network and can diverge from training when upstream moves."
        )


def plan_hub_access(
    configs: Any, environ: Mapping[str, str] | None = None
) -> HubPlan:
    """Decide, from the configs and the local cache alone, whether to go offline."""
    env = os.environ if environ is None else environ
    models = pinned_models(configs)
    return HubPlan(
        models=models,
        missing=tuple(
            model for model in models if model.is_pinned and not snapshot_present(model, env)
        ),
        unpinned=tuple(model for model in models if not model.is_pinned),
    )


def apply_offline_env(
    environ: MutableMapping[str, str] | None = None,
    modules: Mapping[str, Any] | None = None,
) -> None:
    """Set :data:`OFFLINE_ENV` -- before any Hugging Face library is imported.

    Those libraries read the flag once, at import, so setting it afterwards
    would look like offline mode and not be. That is refused rather than
    trusted, unless the flag was already set when they were imported.
    """
    env = os.environ if environ is None else environ
    loaded = sys.modules if modules is None else modules
    imported = [name for name in _HUB_LIBRARIES if name in loaded]
    already_offline = all(_is_true(env.get(name)) for name in OFFLINE_ENV)
    if imported and not already_offline:
        raise RuntimeError(
            f"Offline mode must be chosen before {', '.join(imported)} is imported: "
            "it reads HF_HUB_OFFLINE once, at import time."
        )
    for name in OFFLINE_ENV:
        env[name] = "1"


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().upper() in {"1", "ON", "YES", "TRUE"}


def warm_models(
    configs: Any,
    log: Callable[[str], None] | None = None,
    environ: Mapping[str, str] | None = None,
) -> HubPlan:
    """Fetch exactly the pinned snapshots serving loads, then re-check the cache.

    The download goes through the same three loaders serving uses, at the same
    revisions, so what lands in the cache is precisely the set of files an
    offline load will ask for -- not a whole repository (``whisper-tiny.en``
    also ships TensorFlow and Flax weights nothing here reads).

    Refuses an unpinned revision: warming ``main`` would cache whatever upstream
    serves today and record nothing about which commit that was.
    """
    logger = log or (lambda _message: None)
    env = os.environ if environ is None else environ
    plan = plan_hub_access(configs, env)
    if plan.unpinned:
        raise ValueError(
            plan.unpinned_message() + " Pin them in ml/configs/ before warming."
        )
    if _is_true(env.get("HF_HUB_OFFLINE")) or _is_true(env.get("TRANSFORMERS_OFFLINE")):
        if plan.missing:
            raise RuntimeError(
                "HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE is set, so the missing "
                "snapshot(s) cannot be downloaded: "
                + ", ".join(model.describe() for model in plan.missing)
            )

    from slotify_rank.embeddings import minilm_text, whisper_audio
    from slotify_rank.transcription import local_whisper

    loaders: Sequence[tuple[str, Any]] = (
        ("transcription", local_whisper.WhisperTranscriber(config=configs.transcription)),
        ("embeddings.audio", whisper_audio.WhisperAudioEncoder(config=configs.embeddings.audio)),
        ("embeddings.text", minilm_text.MiniLmTextEncoder(config=configs.embeddings.text)),
    )
    for section, wrapper in loaders:
        config = wrapper.config
        logger(f"  {section}: {config.model_id}@{config.model_revision}")
        wrapper.load()
        wrapper.release()

    after = plan_hub_access(configs, env)
    if after.missing:
        raise SnapshotsMissing(
            "The loaders ran but the cache still lacks "
            + ", ".join(model.describe() for model in after.missing)
            + f" under {hub_cache_dir(env)}. Check HF_HOME / HF_HUB_CACHE."
        )
    return after
