"""Pinned model revisions, the local snapshot check, offline mode and ``infer warm``.

No network and no model weights: the Hugging Face cache is a directory tree in
``tmp_path`` laid out as ``huggingface_hub`` lays it out, and the loaders that
``infer warm`` drives are replaced with fakes that "download" by writing into it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from slotify_rank.cli import main
from slotify_rank.config.feature_settings import (
    load_embedding_config,
    load_transcription_config,
)
from slotify_rank.config.settings import find_repo_root
from slotify_rank.inference.episode import PipelineConfigs
from slotify_rank.inference.hub import (
    COMMIT_SHA,
    OFFLINE_ENV,
    WARM_COMMAND,
    PinnedModel,
    SnapshotsMissing,
    apply_offline_env,
    hub_cache_dir,
    pinned_models,
    plan_hub_access,
    snapshot_dir,
    snapshot_present,
    warm_models,
)
from tests.inference_fakes import (
    FIXTURE_AUDIO,
    SHIPPED_CHECKPOINT,
    install,
    serving_fixtures_missing,
)

CONFIG_DIR = Path(find_repo_root()) / "ml" / "configs"

#: The revisions the committed configs pin. Changing one is a deliberate act
#: that should fail this test and be reviewed: it changes what serving loads.
WHISPER_SHA = "87c7102498dcde7456f24cfd30239ca606ed9063"
MINILM_SHA = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _configs() -> PipelineConfigs:
    return PipelineConfigs.load(CONFIG_DIR)


def _unpinned(configs: PipelineConfigs) -> PipelineConfigs:
    """The same configs with every revision floating on ``main``."""
    return replace(
        configs,
        transcription=replace(configs.transcription, model_revision="main"),
        embeddings=replace(
            configs.embeddings,
            audio=replace(configs.embeddings.audio, model_revision="main"),
            text=replace(configs.embeddings.text, model_revision="main"),
        ),
    )


def _write_snapshot(root: Path, model_id: str, revision: str, weights: bool = True) -> Path:
    directory = root / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (directory / "model.safetensors").write_bytes(b"weights")
    return directory


@pytest.fixture
def hub_cache(tmp_path: Path, monkeypatch) -> Path:
    """An empty hub cache that every lookup in this process resolves to."""
    root = tmp_path / "hub"
    root.mkdir()
    for name in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "XDG_CACHE_HOME",
        *OFFLINE_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    return root


def _fill(root: Path) -> None:
    _write_snapshot(root, "openai/whisper-tiny.en", WHISPER_SHA)
    _write_snapshot(root, "sentence-transformers/all-MiniLM-L6-v2", MINILM_SHA)


# ---------------------------------------------------------------------------
# The committed configs
# ---------------------------------------------------------------------------


def test_the_committed_configs_pin_every_model_to_a_commit_sha() -> None:
    transcription = load_transcription_config(CONFIG_DIR / "transcription_v1.yaml")
    embeddings = load_embedding_config(CONFIG_DIR / "embeddings_v1.yaml")
    assert transcription.model_revision == WHISPER_SHA
    assert embeddings.audio.model_revision == WHISPER_SHA
    assert embeddings.text.model_revision == MINILM_SHA
    assert all(model.is_pinned for model in pinned_models(_configs()))


def test_transcription_and_the_audio_encoder_load_the_same_whisper() -> None:
    # One entry, two users: the transcript and the speech representation come
    # from the same weights.
    whisper = [m for m in pinned_models(_configs()) if m.model_id == "openai/whisper-tiny.en"]
    assert len(whisper) == 1
    assert whisper[0].used_by == ("transcription", "embeddings.audio")


def test_a_disagreeing_whisper_revision_is_listed_twice_not_collapsed() -> None:
    configs = _configs()
    split = replace(
        configs, transcription=replace(configs.transcription, model_revision="a" * 40)
    )
    whisper = [m for m in pinned_models(split) if m.model_id == "openai/whisper-tiny.en"]
    assert [m.used_by for m in whisper] == [("transcription",), ("embeddings.audio",)]


@pytest.mark.parametrize(
    ("revision", "pinned"),
    [
        (WHISPER_SHA, True),
        ("main", False),
        ("v1.0", False),
        (WHISPER_SHA[:12], False),
        (WHISPER_SHA.upper(), False),
    ],
)
def test_only_a_full_commit_sha_counts_as_pinned(revision: str, pinned: bool) -> None:
    assert bool(COMMIT_SHA.match(revision)) is pinned
    model = PinnedModel("org/name", revision, "transformers", ("transcription",))
    assert model.is_pinned is pinned


# ---------------------------------------------------------------------------
# Where the cache is
# ---------------------------------------------------------------------------


def test_hub_cache_resolution_follows_huggingface_hub(tmp_path: Path) -> None:
    home = Path("~").expanduser()
    assert hub_cache_dir({}) == home / ".cache" / "huggingface" / "hub"
    assert hub_cache_dir({"XDG_CACHE_HOME": str(tmp_path / "xdg")}) == (
        tmp_path / "xdg" / "huggingface" / "hub"
    )
    assert hub_cache_dir({"HF_HOME": str(tmp_path / "hf")}) == tmp_path / "hf" / "hub"
    legacy = {"HF_HOME": str(tmp_path / "hf"), "HUGGINGFACE_HUB_CACHE": str(tmp_path / "old")}
    assert hub_cache_dir(legacy) == tmp_path / "old"
    assert hub_cache_dir({**legacy, "HF_HUB_CACHE": str(tmp_path / "new")}) == tmp_path / "new"


def test_each_loader_honours_its_own_cache_override(tmp_path: Path) -> None:
    whisper = PinnedModel("openai/whisper-tiny.en", WHISPER_SHA, "transformers", ())
    minilm = PinnedModel(
        "sentence-transformers/all-MiniLM-L6-v2", MINILM_SHA, "sentence-transformers", ()
    )
    env = {
        "HF_HUB_CACHE": str(tmp_path / "hub"),
        "TRANSFORMERS_CACHE": str(tmp_path / "tf"),
        "SENTENCE_TRANSFORMERS_HOME": str(tmp_path / "st"),
    }
    assert snapshot_dir(whisper, env).is_relative_to(tmp_path / "tf")
    assert snapshot_dir(minilm, env).is_relative_to(tmp_path / "st")
    assert snapshot_dir(whisper, {"HF_HUB_CACHE": str(tmp_path / "hub")}) == (
        tmp_path / "hub" / "models--openai--whisper-tiny.en" / "snapshots" / WHISPER_SHA
    )


def test_a_snapshot_needs_its_config_and_weights(tmp_path: Path) -> None:
    model = PinnedModel("openai/whisper-tiny.en", WHISPER_SHA, "transformers", ())
    env = {"HF_HUB_CACHE": str(tmp_path)}
    assert not snapshot_present(model, env)
    _write_snapshot(tmp_path, model.model_id, WHISPER_SHA, weights=False)
    assert not snapshot_present(model, env)
    _write_snapshot(tmp_path, model.model_id, WHISPER_SHA)
    assert snapshot_present(model, env)


def test_an_unpinned_revision_is_never_present_even_if_main_is_cached(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, "openai/whisper-tiny.en", "main")
    model = PinnedModel("openai/whisper-tiny.en", "main", "transformers", ())
    assert not snapshot_present(model, {"HF_HUB_CACHE": str(tmp_path)})


# ---------------------------------------------------------------------------
# The offline decision
# ---------------------------------------------------------------------------


def test_offline_when_every_pinned_snapshot_is_local(hub_cache: Path) -> None:
    _fill(hub_cache)
    plan = plan_hub_access(_configs())
    assert plan.offline
    assert plan.missing == () and plan.unpinned == ()


def test_a_missing_snapshot_blocks_offline_and_names_the_warm_command(hub_cache: Path) -> None:
    _write_snapshot(hub_cache, "openai/whisper-tiny.en", WHISPER_SHA)
    plan = plan_hub_access(_configs())
    assert not plan.offline
    assert [m.model_id for m in plan.missing] == ["sentence-transformers/all-MiniLM-L6-v2"]
    message = plan.missing_message()
    # The API shows a client the first ~300 characters of an error line: the
    # fix must be inside them, ahead of the SHAs.
    assert WARM_COMMAND in message[:300]
    assert MINILM_SHA in message


def test_unpinned_revisions_are_never_offline(hub_cache: Path) -> None:
    _fill(hub_cache)
    plan = plan_hub_access(_unpinned(_configs()))
    assert not plan.offline
    assert plan.missing == ()
    assert len(plan.unpinned) == 2
    assert "not pinned" in plan.unpinned_message()


def test_offline_mode_sets_both_variables() -> None:
    env: dict[str, str] = {}
    apply_offline_env(env, modules={})
    assert env == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_offline_mode_is_refused_once_a_hub_library_is_imported() -> None:
    # They read the flag at import: setting it afterwards would only look offline.
    with pytest.raises(RuntimeError, match="before transformers is imported"):
        apply_offline_env({}, modules={"transformers": object()})


def test_offline_mode_is_accepted_when_it_was_already_on_at_import() -> None:
    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    apply_offline_env(env, modules={"huggingface_hub": object()})
    assert env["HF_HUB_OFFLINE"] == "1"


# ---------------------------------------------------------------------------
# Revision pinning and the shipped checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.skipif(bool(serving_fixtures_missing()), reason=serving_fixtures_missing() or "ok")
def test_pinning_changes_no_value_the_checkpoint_compatibility_check_compares(
    tmp_path: Path, monkeypatch
) -> None:
    """``main`` -> its own SHA changes cache identities and config digests only.

    What ``assert_compatible`` and ``align_examples`` compare -- handcrafted
    names, block widths and parts, feature spec / pipeline / input-schema
    versions, the normalizer's identity -- is built from code constants and the
    extracted features, never from a model revision. So the served schema is
    identical either way, and so is the ranking.
    """
    install(monkeypatch)

    from slotify_rank.inference.episode import prepare_episode, workspace
    from slotify_rank.inference.predictor import RankerPredictor
    from slotify_rank.inference.service import rank_audio

    pinned = _configs()
    floating = _unpinned(pinned)
    assert pinned.embeddings.digest != floating.embeddings.digest
    assert pinned.transcription.digest != floating.transcription.digest

    schemas = []
    for name, configs in (("pinned", pinned), ("floating", floating)):
        with workspace(tmp_path / name) as space:
            schemas.append(prepare_episode(space, FIXTURE_AUDIO, configs).dataset.schema)
    assert schemas[0] == schemas[1]

    predictor = RankerPredictor.load(SHIPPED_CHECKPOINT)
    documents = []
    for name, configs in (("pinned-rank", pinned), ("floating-rank", floating)):
        document = rank_audio(predictor, configs, FIXTURE_AUDIO, tmp_path / name).to_dict()
        document.pop("timings_seconds")
        documents.append(document)
    assert documents[0] == documents[1]
    assert documents[0]["ranked"], "the fixture should produce candidates"


# ---------------------------------------------------------------------------
# infer warm
# ---------------------------------------------------------------------------


def _fake_loaders(monkeypatch, cache: Path) -> list[str]:
    """Loaders that 'download' their pinned snapshot into ``cache``."""
    from slotify_rank.embeddings import minilm_text, whisper_audio
    from slotify_rank.transcription import local_whisper

    loaded: list[str] = []

    def fake(name: str):
        class Loader:
            def __init__(self, config):
                self.config = config

            def load(self):
                loaded.append(f"{name}:{self.config.model_id}@{self.config.model_revision}")
                _write_snapshot(cache, self.config.model_id, self.config.model_revision)
                return self

            def release(self):
                return None

        return Loader

    monkeypatch.setattr(local_whisper, "WhisperTranscriber", fake("transcription"))
    monkeypatch.setattr(whisper_audio, "WhisperAudioEncoder", fake("audio"))
    monkeypatch.setattr(minilm_text, "MiniLmTextEncoder", fake("text"))
    return loaded


def test_warm_loads_exactly_the_pinned_revisions(hub_cache: Path, monkeypatch) -> None:
    loaded = _fake_loaders(monkeypatch, hub_cache)
    plan = warm_models(_configs())
    assert plan.offline
    assert loaded == [
        f"transcription:openai/whisper-tiny.en@{WHISPER_SHA}",
        f"audio:openai/whisper-tiny.en@{WHISPER_SHA}",
        f"text:sentence-transformers/all-MiniLM-L6-v2@{MINILM_SHA}",
    ]


def test_warm_refuses_an_unpinned_revision(hub_cache: Path, monkeypatch) -> None:
    loaded = _fake_loaders(monkeypatch, hub_cache)
    with pytest.raises(ValueError, match="not pinned"):
        warm_models(_unpinned(_configs()))
    assert loaded == []


def test_warm_cannot_download_in_offline_mode(hub_cache: Path, monkeypatch) -> None:
    loaded = _fake_loaders(monkeypatch, hub_cache)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="cannot be downloaded"):
        warm_models(_configs())
    assert loaded == []


def test_warm_fails_if_the_loaders_leave_the_cache_incomplete(
    hub_cache: Path, monkeypatch
) -> None:
    _fake_loaders(monkeypatch, hub_cache / "somewhere-else")
    with pytest.raises(SnapshotsMissing, match="still lacks"):
        warm_models(_configs())


def test_warm_check_reports_and_exits_by_presence(hub_cache: Path, capsys) -> None:
    assert main(["infer", "warm", "--check"]) == 1
    captured = capsys.readouterr()
    assert "missing" in captured.out
    assert WARM_COMMAND in captured.err

    _fill(hub_cache)
    assert main(["infer", "warm", "--check"]) == 0
    captured = capsys.readouterr()
    assert f"openai/whisper-tiny.en@{WHISPER_SHA}" in captured.out
    assert "run offline" in captured.out


def test_warm_command_downloads_then_reports_present(hub_cache: Path, monkeypatch, capsys) -> None:
    loaded = _fake_loaders(monkeypatch, hub_cache)
    assert main(["infer", "warm"]) == 0
    assert len(loaded) == 3
    assert "present at" in capsys.readouterr().out
