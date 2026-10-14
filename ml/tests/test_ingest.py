"""Import, fetch, probe and normalization-cache behaviour.

The fetch tests stub the HTTP layer -- no test in this suite reaches the network.
The probe/normalize tests need real FFmpeg and skip cleanly without it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from slotify_rank.config.settings import find_repo_root
from slotify_rank.config.versions import PREPROCESSING_VERSION
from slotify_rank.data import fetch as fetch_module
from slotify_rank.data.checksum import ChecksumMismatch, sha256_file
from slotify_rank.data.ffmpeg import FFmpegNotFound, require_ffmpeg
from slotify_rank.data.import_local import import_source
from slotify_rank.data.normalize import (
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE_HZ,
    needs_normalization,
    normalize_episode,
)
from slotify_rank.data.paths import DataPaths
from slotify_rank.data.probe import UnreadableAudio, probe_audio
from slotify_rank.data.sources import SourceEntry

from tests.dataset_fixtures import make_episode, write_speech_like_wav


def _has_ffmpeg() -> bool:
    try:
        require_ffmpeg()
    except FFmpegNotFound:
        return False
    return True


needs_ffmpeg = pytest.mark.skipif(not _has_ffmpeg(), reason="FFmpeg is not installed")


@pytest.fixture()
def paths(tmp_path: Path) -> DataPaths:
    instance = DataPaths(repo_root=tmp_path, data_root=tmp_path / "data")
    instance.mkdirs()
    return instance


def _local_entry(path: Path, **overrides) -> SourceEntry:
    payload = {
        "id": "local-one",
        "source_type": "local_file",
        "title": "Local one",
        "series_id": "local-series",
        "source_name": "Private",
        "content_type": "podcast",
        "path": str(path),
    }
    payload.update(overrides)
    return SourceEntry(**payload)


# --------------------------------------------------------------------------
# Local import
# --------------------------------------------------------------------------


def test_local_file_is_copied_into_the_data_root(paths, tmp_path: Path):
    origin = tmp_path / "outside" / "clip.mp3"
    origin.parent.mkdir()
    origin.write_bytes(b"pretend mp3 bytes")

    result = import_source(_local_entry(origin), paths)

    assert result.copied is True
    assert result.episode.original_path.startswith("data/raw/")
    copied = paths.absolute(result.episode.original_path)
    assert copied.read_bytes() == b"pretend mp3 bytes"
    assert result.episode.sha256 == sha256_file(origin)


def test_local_import_is_idempotent(paths, tmp_path: Path):
    origin = tmp_path / "clip.mp3"
    origin.write_bytes(b"pretend mp3 bytes")
    first = import_source(_local_entry(origin), paths)
    second = import_source(_local_entry(origin), paths)
    assert first.episode.episode_id == second.episode.episode_id
    assert second.copied is False


def test_local_import_preserves_provenance_and_privacy(paths, tmp_path: Path):
    origin = tmp_path / "clip.mp3"
    origin.write_bytes(b"pretend mp3 bytes")
    episode = import_source(_local_entry(origin), paths).episode
    assert episode.source_type == "local_file"
    assert episode.source_uri == str(origin)
    assert episode.license_name is None, "a private file gets no assumed licence"


def test_local_import_refuses_to_overwrite_different_content(paths, tmp_path: Path):
    origin = tmp_path / "clip.mp3"
    origin.write_bytes(b"first content")
    result = import_source(_local_entry(origin), paths)
    destination = paths.absolute(result.episode.original_path)
    destination.write_bytes(b"tampered content")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        import_source(_local_entry(origin), paths)


def test_repository_fixture_is_referenced_not_copied(paths, tmp_path: Path):
    fixture = paths.repo_root / "backend" / "audio_tests" / "x.mp3"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(b"fixture bytes")
    result = import_source(
        _local_entry(
            fixture, source_type="existing_repository_fixture", id="fixture-one"
        ),
        paths,
    )
    assert result.copied is False
    assert result.episode.original_path == "backend/audio_tests/x.mp3"


def test_fixture_outside_the_repository_is_rejected(paths):
    # `paths.repo_root` is the scratch directory, so "outside" must be a
    # different scratch directory entirely.
    import tempfile

    with tempfile.TemporaryDirectory() as elsewhere:
        outside = Path(elsewhere) / "x.mp3"
        outside.write_bytes(b"bytes")
        with pytest.raises(ValueError, match="outside the repository"):
            import_source(
                _local_entry(outside, source_type="existing_repository_fixture"), paths
            )


def test_missing_local_file_is_reported_clearly(paths, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="audio file not found"):
        import_source(_local_entry(tmp_path / "nope.mp3"), paths)


def test_unrecognised_extension_is_rejected(paths, tmp_path: Path):
    origin = tmp_path / "clip.txt"
    origin.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="recognised audio extension"):
        import_source(_local_entry(origin), paths)


def test_import_preserves_already_established_metadata(paths, tmp_path: Path):
    """Re-importing must not reset an episode that has already been normalized."""
    origin = tmp_path / "clip.mp3"
    origin.write_bytes(b"pretend mp3 bytes")
    first = import_source(_local_entry(origin), paths).episode
    advanced = first.replace(
        status="normalized",
        normalized_path=f"data/normalized/{first.episode_id}.wav",
        normalized_sha256=first.sha256,
        duration_ms=12_345,
        preprocessing_version=PREPROCESSING_VERSION,
    )
    again = import_source(
        _local_entry(origin), paths, {advanced.episode_id: advanced}
    ).episode
    assert again.status == "normalized"
    assert again.duration_ms == 12_345


# --------------------------------------------------------------------------
# Fetch (network stubbed)
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def read(self, size: int) -> bytes:
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def stub_download(monkeypatch):
    def install(payload: bytes):
        monkeypatch.setattr(
            fetch_module.urllib.request,
            "urlopen",
            lambda request, timeout=None: _FakeResponse(payload),
        )

    return install


def test_download_verifies_a_matching_checksum(tmp_path: Path, stub_download):
    payload = b"remote audio bytes"
    stub_download(payload)
    expected = hashlib.sha256(payload).hexdigest()
    digest, written, downloaded = fetch_module.download_to(
        "https://example.org/a.mp3", tmp_path / "a.mp3", expected
    )
    assert (digest, written, downloaded) == (expected, len(payload), True)


def test_download_rejects_a_checksum_mismatch(tmp_path: Path, stub_download):
    stub_download(b"different bytes")
    destination = tmp_path / "a.mp3"
    with pytest.raises(ChecksumMismatch):
        fetch_module.download_to("https://example.org/a.mp3", destination, "0" * 64)
    assert not destination.exists(), "a rejected download must leave nothing behind"


def test_download_leaves_no_partial_file_on_failure(tmp_path: Path, monkeypatch):
    def explode(request, timeout=None):
        raise fetch_module.urllib.error.URLError("network down")

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", explode)
    destination = tmp_path / "a.mp3"
    with pytest.raises(fetch_module.FetchError, match="Could not download"):
        fetch_module.download_to("https://example.org/a.mp3", destination)
    assert list(tmp_path.iterdir()) == []


def test_download_reuses_an_existing_matching_file(tmp_path: Path, stub_download):
    payload = b"remote audio bytes"
    destination = tmp_path / "a.mp3"
    destination.write_bytes(payload)
    stub_download(b"should not be read")
    digest, _, downloaded = fetch_module.download_to(
        "https://example.org/a.mp3", destination, hashlib.sha256(payload).hexdigest()
    )
    assert downloaded is False
    assert digest == hashlib.sha256(payload).hexdigest()


def test_existing_file_with_a_different_checksum_is_not_replaced(
    tmp_path: Path, stub_download
):
    destination = tmp_path / "a.mp3"
    destination.write_bytes(b"local content")
    stub_download(b"remote content")
    with pytest.raises(ChecksumMismatch):
        fetch_module.download_to(
            "https://example.org/a.mp3", destination, "0" * 64
        )
    assert destination.read_bytes() == b"local content"


def test_zero_byte_download_is_rejected(tmp_path: Path, stub_download):
    stub_download(b"")
    with pytest.raises(fetch_module.FetchError, match="zero bytes"):
        fetch_module.download_to("https://example.org/a.mp3", tmp_path / "a.mp3")


# --------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------


@needs_ffmpeg
def test_probe_reads_real_metadata(tmp_path: Path):
    path = write_speech_like_wav(tmp_path / "a.wav", [(1500, True), (500, False)])
    metadata = probe_audio(path)
    assert 1900 <= metadata.duration_ms <= 2100
    assert metadata.sample_rate_hz == 16_000
    assert metadata.channels == 1
    assert metadata.file_format == "wav"


def test_probe_rejects_a_zero_length_file(tmp_path: Path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(UnreadableAudio, match="zero-length"):
        probe_audio(path)


@needs_ffmpeg
def test_probe_rejects_a_non_audio_file(tmp_path: Path):
    path = tmp_path / "notes.wav"
    path.write_text("this is not audio at all", encoding="utf-8")
    with pytest.raises(UnreadableAudio):
        probe_audio(path)


def test_probe_rejects_a_missing_file(tmp_path: Path):
    with pytest.raises(UnreadableAudio, match="Not a readable file"):
        probe_audio(tmp_path / "nope.wav")


def test_missing_ffmpeg_gives_an_actionable_message(monkeypatch):
    import slotify_rank.data.ffmpeg as ffmpeg_module

    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    with pytest.raises(FFmpegNotFound) as error:
        ffmpeg_module.ffmpeg_path()
    message = str(error.value)
    assert "not found on PATH" in message
    assert "FFMPEG_BIN" in message


# --------------------------------------------------------------------------
# Normalization and its cache
# --------------------------------------------------------------------------


def _importable_source(paths: DataPaths, tmp_path: Path):
    origin = tmp_path / "source.wav"
    write_speech_like_wav(origin, [(1500, True), (700, False), (1500, True)])
    return import_source(_local_entry(origin), paths).episode


@needs_ffmpeg
def test_normalize_renders_to_the_ml_format(paths, tmp_path: Path):
    episode = _importable_source(paths, tmp_path)
    result = normalize_episode(episode, paths)
    assert result.rendered is True
    normalized = result.episode
    assert normalized.status == "normalized"
    assert normalized.preprocessing_version == PREPROCESSING_VERSION
    metadata = probe_audio(paths.absolute(normalized.normalized_path))
    assert metadata.sample_rate_hz == TARGET_SAMPLE_RATE_HZ
    assert metadata.channels == TARGET_CHANNELS


@needs_ffmpeg
def test_normalize_does_not_render_an_unchanged_file_twice(paths, tmp_path: Path):
    episode = _importable_source(paths, tmp_path)
    first = normalize_episode(episode, paths).episode
    second = normalize_episode(first, paths)
    assert second.rendered is False
    assert needs_normalization(first, paths) is False


@needs_ffmpeg
def test_a_preprocessing_version_bump_invalidates_the_cache(paths, tmp_path: Path):
    episode = _importable_source(paths, tmp_path)
    normalized = normalize_episode(episode, paths).episode
    stale = normalized.replace(preprocessing_version="preprocess-v0.9.0")
    assert needs_normalization(stale, paths) is True
    assert normalize_episode(stale, paths).rendered is True


@needs_ffmpeg
def test_a_tampered_render_invalidates_the_cache(paths, tmp_path: Path):
    episode = _importable_source(paths, tmp_path)
    normalized = normalize_episode(episode, paths).episode
    target = paths.absolute(normalized.normalized_path)
    write_speech_like_wav(target, [(500, True)])
    assert needs_normalization(normalized, paths) is True


@needs_ffmpeg
def test_force_re_renders(paths, tmp_path: Path):
    episode = _importable_source(paths, tmp_path)
    normalized = normalize_episode(episode, paths).episode
    assert normalize_episode(normalized, paths, force=True).rendered is True


def test_normalize_refuses_when_the_source_hash_has_changed(paths, tmp_path: Path):
    episode = _importable_source(paths, tmp_path)
    source = paths.absolute(episode.original_path)
    source.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="no longer matches its recorded SHA-256"):
        normalize_episode(episode, paths)


def test_normalize_reports_a_missing_original(paths):
    episode = make_episode(status="probed", normalized_path=None)
    with pytest.raises(FileNotFoundError, match="original audio missing"):
        normalize_episode(episode, paths)


# --------------------------------------------------------------------------
# A data root outside the repository (SLOTIFY_DATA_ROOT / --data-root)
# --------------------------------------------------------------------------


@pytest.fixture()
def external(tmp_path: Path) -> DataPaths:
    """A repository and a data root that are siblings, not nested."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    instance = DataPaths(repo_root=repo, data_root=tmp_path / "corpus")
    instance.mkdirs()
    return instance


def test_an_external_data_root_stores_data_root_relative_paths(external):
    from slotify_rank.data.paths import DATA_ROOT_PREFIX

    raw = external.raw_dir / "clip.mp3"
    stored = external.relative(raw)
    assert stored == f"{DATA_ROOT_PREFIX}raw/clip.mp3"
    assert external.absolute(stored) == raw.resolve()
    # A file inside the repository is still stored repository-relative.
    fixture = external.repo_root / "ml" / "tests" / "fixtures" / "a.wav"
    assert external.relative(fixture) == "ml/tests/fixtures/a.wav"


def test_a_data_root_inside_the_repository_is_stored_exactly_as_before(paths):
    raw = paths.raw_dir / "clip.mp3"
    assert paths.relative(raw) == "data/raw/clip.mp3"
    assert paths.absolute("data/raw/clip.mp3") == raw.resolve()


def test_a_path_outside_both_roots_is_still_refused(external, tmp_path: Path):
    with pytest.raises(ValueError, match="outside the repository root"):
        external.relative(tmp_path / "elsewhere" / "clip.mp3")


def test_data_root_relative_paths_reject_traversal(external):
    with pytest.raises(ValueError, match="traverse upward"):
        external.absolute("@data-root/../secrets")


def test_data_root_relative_paths_follow_a_moved_corpus(external, tmp_path: Path):
    """The point of storing them relatively: the drive can be mounted elsewhere."""
    stored = external.relative(external.normalized_dir / "ep.wav")
    moved = DataPaths(repo_root=external.repo_root, data_root=tmp_path / "mounted")
    assert moved.absolute(stored) == (tmp_path / "mounted" / "normalized" / "ep.wav")


def test_the_environment_variable_resolves_data_root_paths(
    external, tmp_path: Path, monkeypatch
):
    from slotify_rank.data.paths import resolve

    monkeypatch.setenv("SLOTIFY_DATA_ROOT", str(tmp_path / "corpus"))
    assert resolve("@data-root/raw/x.mp3", external.repo_root) == (
        tmp_path / "corpus" / "raw" / "x.mp3"
    ).resolve()


def test_import_into_an_external_data_root(external, tmp_path: Path):
    origin = tmp_path / "outside" / "clip.mp3"
    origin.parent.mkdir()
    origin.write_bytes(b"pretend mp3 bytes")

    episode = import_source(_local_entry(origin), external).episode

    assert episode.original_path.startswith("@data-root/raw/")
    assert external.absolute(episode.original_path).read_bytes() == b"pretend mp3 bytes"


def test_fetch_into_an_external_data_root(external, stub_download):
    from slotify_rank.data.sources import SourceEntry as Entry

    stub_download(b"remote audio bytes")
    entry = Entry(
        id="remote-one",
        source_type="direct_download",
        url="https://example.org/remote-one.mp3",
        title="Remote one",
        series_id="remote-series",
        source_name="Example",
        content_type="podcast",
        license_name="CC0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    )
    episode = fetch_module.fetch_source(entry, external).episode
    assert episode.original_path == "@data-root/raw/remote-one.mp3"
    assert external.absolute(episode.original_path).is_file()


def test_validation_accepts_an_external_data_root(external, tmp_path: Path):
    from slotify_rank.data.validate import validate_dataset

    origin = tmp_path / "outside" / "clip.mp3"
    origin.parent.mkdir()
    origin.write_bytes(b"pretend mp3 bytes")
    episode = import_source(_local_entry(origin), external).episode

    report = validate_dataset([episode], [], external, deep=True)
    path_errors = [
        finding
        for finding in report.errors
        if finding.check
        in ("episode_paths_within_data_roots", "episode_absolute_paths", "episode_checksums")
    ]
    assert path_errors == []


@needs_ffmpeg
def test_normalize_into_an_external_data_root(external, tmp_path: Path):
    """Used to render the WAV and *then* raise, leaving an orphaned render."""
    origin = tmp_path / "outside" / "clip.wav"
    write_speech_like_wav(origin, [(1500, True), (800, False), (1500, True)])
    episode = import_source(_local_entry(origin), external).episode
    episode = episode.replace(
        status="probed", duration_ms=probe_audio(origin).duration_ms
    )

    result = normalize_episode(episode, external)

    assert result.episode.normalized_path.startswith("@data-root/normalized/")
    assert external.absolute(result.episode.normalized_path).is_file()
    assert needs_normalization(result.episode, external) is False
