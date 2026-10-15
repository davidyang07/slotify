"""``infer serve``: the persistent worker, its protocol, its scratch discipline.

Four layers, cheapest first:

* the resident-model registry and the three model wrappers' use of it, with
  ``transformers`` / ``sentence_transformers`` replaced by stub modules;
* request validation and the serve loop, with a stub handler and in-memory
  streams;
* the real :class:`RankerWorker` over the shipped checkpoint and a real
  repository fixture, with the three pretrained models faked
  (``tests/inference_fakes.py``) -- compared document-for-document with the
  one-shot ``infer rank``;
* the real ``python -m slotify_rank.cli infer serve`` process, for what only a
  process can show: stdout discipline, SIGTERM, startup refusal.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

from slotify_rank.cli import main
from slotify_rank.embeddings import resident
from slotify_rank.inference.episode import PipelineConfigs
from slotify_rank.inference.hub import WARM_COMMAND
from slotify_rank.inference.service import (
    Terminated,
    request_scratch,
    terminate_on_sigterm,
)
from slotify_rank.inference.worker import (
    RankerWorker,
    RequestError,
    WorkerRequest,
    last_meaningful_line,
    preload_models,
    serve,
)
from tests.inference_fakes import (
    BOOTSTRAP,
    FIXTURE_AUDIO,
    LOADS,
    SHIPPED_CHECKPOINT,
    install,
    serving_fixtures_missing,
)

ML_DIR = Path(__file__).resolve().parents[1]

needs_serving_fixtures = pytest.mark.skipif(
    bool(serving_fixtures_missing()), reason=serving_fixtures_missing() or "available"
)


# ---------------------------------------------------------------------------
# Resident models
# ---------------------------------------------------------------------------


def test_outside_a_scope_every_acquire_builds() -> None:
    built: list[int] = []
    for _ in range(2):
        resident.acquire("key", lambda: built.append(1) or object())
    assert len(built) == 2
    assert resident.resident_count() == 0
    assert not resident.is_open()


def test_inside_a_scope_each_key_is_built_once() -> None:
    built: list[str] = []

    def build(name: str):
        def make():
            built.append(name)
            return (object(), object())

        return make

    with resident.resident_models():
        first = resident.acquire(("a", 1), build("a"))
        again = resident.acquire(("a", 1), build("a"))
        other = resident.acquire(("a", 2), build("b"))
        assert first is again and other is not first
        assert built == ["a", "b"]
        assert resident.resident_count() == 2
        # Held tuple members count; an equal-but-different object does not.
        assert resident.is_resident(first[1])
        assert not resident.is_resident(object())
    assert resident.resident_count() == 0
    assert not resident.is_resident(first[1])


def test_scopes_do_not_nest() -> None:
    with resident.resident_models():
        with pytest.raises(RuntimeError, match="already open"):
            with resident.resident_models():
                pass


class _Weights:
    """Stands in for a model object: eval()/to()/get_encoder() return itself."""

    def __init__(self, calls: list[str], name: str) -> None:
        calls.append(name)
        self.max_seq_length = 0

    def eval(self):
        return self

    def to(self, _device):
        return self

    def get_encoder(self):
        return self


@pytest.fixture
def stub_hub_libraries(monkeypatch) -> list[str]:
    """``transformers`` and ``sentence_transformers`` whose loaders only count."""
    calls: list[str] = []

    def loader(name: str):
        return types.SimpleNamespace(
            from_pretrained=lambda model_id, **_kwargs: _Weights(calls, name)
        )

    transformers = types.ModuleType("transformers")
    for name in (
        "WhisperProcessor",
        "WhisperForConditionalGeneration",
        "WhisperFeatureExtractor",
        "WhisperModel",
    ):
        setattr(transformers, name, loader(name))
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = (  # type: ignore[attr-defined]
        lambda model_id, **_kwargs: _Weights(calls, "SentenceTransformer")
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    return calls


def _wrappers(configs: PipelineConfigs):
    from slotify_rank.embeddings.minilm_text import MiniLmTextEncoder
    from slotify_rank.embeddings.whisper_audio import WhisperAudioEncoder
    from slotify_rank.transcription.local_whisper import WhisperTranscriber

    return (
        WhisperTranscriber(config=configs.transcription),
        WhisperAudioEncoder(config=configs.embeddings.audio),
        MiniLmTextEncoder(config=configs.embeddings.text),
    )


def test_the_model_wrappers_reload_per_stage_outside_a_scope(stub_hub_libraries, monkeypatch) -> None:
    """The corpus pipeline's behaviour is unchanged: load, use, free, reload."""
    freed: list[object] = []
    monkeypatch.setattr("slotify_rank.embeddings.device.free_model", freed.append)
    configs = PipelineConfigs.load()
    for _ in range(2):
        for wrapper in _wrappers(configs):
            wrapper.load()
            wrapper.release()
    assert len(stub_hub_libraries) == 2 * 5  # processor+model, extractor+model, ST
    assert len(freed) == 2 * 3


def test_inside_a_scope_the_model_wrappers_load_once_and_release_keeps_them(
    stub_hub_libraries, monkeypatch
) -> None:
    freed: list[object] = []
    monkeypatch.setattr("slotify_rank.embeddings.device.free_model", freed.append)
    configs = PipelineConfigs.load()
    with resident.resident_models():
        for _ in range(3):
            for wrapper in _wrappers(configs):
                wrapper.load()
                wrapper.release()
        assert resident.resident_count() == 3
        assert freed == []
    assert sorted(stub_hub_libraries) == sorted(
        [
            "WhisperProcessor",
            "WhisperForConditionalGeneration",
            "WhisperFeatureExtractor",
            "WhisperModel",
            "SentenceTransformer",
        ]
    )


def test_a_different_revision_is_a_different_resident_model(stub_hub_libraries) -> None:
    from dataclasses import replace

    from slotify_rank.transcription.local_whisper import WhisperTranscriber

    configs = PipelineConfigs.load()
    other = replace(configs.transcription, model_revision="b" * 40)
    with resident.resident_models():
        WhisperTranscriber(config=configs.transcription).load()
        WhisperTranscriber(config=other).load()
        assert resident.resident_count() == 2


def test_minilm_max_seq_length_is_applied_on_every_load(stub_hub_libraries) -> None:
    from dataclasses import replace

    from slotify_rank.embeddings.minilm_text import MiniLmTextEncoder

    configs = PipelineConfigs.load()
    shorter = replace(configs.embeddings.text, max_seq_length=64)
    with resident.resident_models():
        first = MiniLmTextEncoder(config=configs.embeddings.text).load()
        assert first._model.max_seq_length == configs.embeddings.text.max_seq_length
        second = MiniLmTextEncoder(config=shorter).load()
        assert second._model is first._model
        assert second._model.max_seq_length == 64


def test_preload_refuses_to_run_outside_a_scope(monkeypatch) -> None:
    install(monkeypatch)
    with pytest.raises(RuntimeError, match="outside resident_models"):
        preload_models(PipelineConfigs.load())


def test_preload_loads_all_three_models(monkeypatch) -> None:
    install(monkeypatch)
    with resident.resident_models():
        preload_models(PipelineConfigs.load())
    assert LOADS == {"transcriber": 1, "audio_encoder": 1, "text_encoder": 1}


# ---------------------------------------------------------------------------
# Scratch files and SIGTERM
# ---------------------------------------------------------------------------


def test_request_scratch_lives_inside_the_work_dir_and_aims_tempfile_there(
    tmp_path: Path, monkeypatch
) -> None:
    system_temp = tmp_path / "system-temp"
    system_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_temp))
    work = tmp_path / "work"
    work.mkdir()
    with request_scratch(work) as scratch:
        assert scratch.parent == work and scratch.is_dir()
        assert Path(tempfile.gettempdir()) == work
        assert Path(tempfile.mkdtemp()).parent == work
    assert tempfile.tempdir == str(system_temp)
    assert list(system_temp.iterdir()) == []


def test_request_scratch_requires_its_owner_to_have_created_the_work_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="its owner must create it"):
        with request_scratch(tmp_path / "absent"):
            pass


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_becomes_an_exception_inside_the_block_only() -> None:
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(Terminated) as caught:
        with terminate_on_sigterm():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)  # interrupted by the handler
    assert caught.value.signum == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is before


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def _line(directory: Path, **fields) -> str:
    body = {"id": "r1", "audio_path": "/audio/x.mp3", "work_dir": str(directory)}
    body.update(fields)
    return json.dumps(body)


def test_a_valid_request_parses_with_defaults(tmp_path: Path) -> None:
    request = WorkerRequest.from_line(_line(tmp_path))
    assert request.id == "r1"
    assert request.work_dir == tmp_path
    assert request.episode_id == "upload"
    assert request.skip_transcription is False
    full = WorkerRequest.from_line(
        _line(tmp_path, id=7, episode_id="ep_1.a-b", skip_transcription=True)
    )
    assert (full.id, full.episode_id, full.skip_transcription) == (7, "ep_1.a-b", True)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"skip_transcribe": True}, "unknown request field"),
        ({"audio_path": ""}, "audio_path"),
        ({"audio_path": 3}, "audio_path"),
        ({"work_dir": None}, "work_dir"),
        ({"work_dir": "relative/dir"}, "absolute path"),
        ({"episode_id": "../escape"}, "episode_id"),
        ({"episode_id": "a/b"}, "episode_id"),
        ({"episode_id": ""}, "episode_id"),
        ({"skip_transcription": "yes"}, "true or false"),
    ],
)
def test_an_invalid_request_is_refused_and_keeps_its_id(tmp_path: Path, fields, message) -> None:
    with pytest.raises(RequestError, match=message) as caught:
        WorkerRequest.from_line(_line(tmp_path, **fields))
    assert caught.value.request_id == "r1"


def test_a_work_dir_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RequestError, match="not an existing directory"):
        WorkerRequest.from_line(_line(tmp_path / "gone"))


@pytest.mark.parametrize(
    "line",
    ["not json", "[1, 2]", '"text"', '{"audio_path": "/a"}', '{"id": true}', '{"id": ""}'],
)
def test_a_request_without_a_readable_id_is_refused_with_a_null_id(line: str) -> None:
    with pytest.raises(RequestError) as caught:
        WorkerRequest.from_line(line)
    assert caught.value.request_id is None


def test_last_meaningful_line() -> None:
    assert last_meaningful_line("Refusing:\n  - a\n  - b\n\n") == "- b"
    assert last_meaningful_line("   ") == "unknown error"


# ---------------------------------------------------------------------------
# The serve loop
# ---------------------------------------------------------------------------


def _serve(lines: list[str], handle) -> tuple[int, list[dict]]:
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    stdout = io.StringIO()
    code = serve(handle, stdin, stdout)
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    return code, responses


def test_each_request_gets_exactly_one_response_in_order(tmp_path: Path) -> None:
    seen: list[str | int] = []

    def handle(request: WorkerRequest):
        seen.append(request.id)
        return {"echo": request.id}

    code, responses = _serve(
        [_line(tmp_path, id="a"), "", _line(tmp_path, id=2), _line(tmp_path, id="c")], handle
    )
    assert code == 0
    assert seen == ["a", 2, "c"]
    assert responses == [
        {"id": "a", "ok": True, "payload": {"echo": "a"}},
        {"id": 2, "ok": True, "payload": {"echo": 2}},
        {"id": "c", "ok": True, "payload": {"echo": "c"}},
    ]


def test_an_expected_failure_is_answered_and_serving_continues(tmp_path: Path) -> None:
    def handle(request: WorkerRequest):
        if request.id == "bad":
            raise RuntimeError("Refusing to load an incompatible checkpoint:\n  - width 3 != 4")
        return {"fine": True}

    code, responses = _serve([_line(tmp_path, id="bad"), _line(tmp_path, id="ok")], handle)
    assert code == 0
    assert responses[0] == {"id": "bad", "ok": False, "error": "- width 3 != 4"}
    assert responses[1]["ok"] is True


def test_an_invalid_line_is_answered_and_serving_continues(tmp_path: Path) -> None:
    code, responses = _serve(["{nope", _line(tmp_path, id="ok")], lambda _r: {"fine": True})
    assert code == 0
    assert responses[0]["id"] is None and responses[0]["ok"] is False
    assert "not JSON" in responses[0]["error"]
    assert responses[1] == {"id": "ok", "ok": True, "payload": {"fine": True}}


def test_an_unexpected_failure_is_answered_then_the_worker_exits(tmp_path: Path) -> None:
    handled: list[str | int] = []

    def handle(request: WorkerRequest):
        handled.append(request.id)
        raise ZeroDivisionError("division by zero")

    code, responses = _serve([_line(tmp_path, id="x"), _line(tmp_path, id="never")], handle)
    assert code == 1
    assert handled == ["x"]
    assert responses == [{"id": "x", "ok": False, "error": "ZeroDivisionError: division by zero"}]


def test_termination_propagates_unanswered(tmp_path: Path) -> None:
    def handle(_request: WorkerRequest):
        raise Terminated(signal.SIGTERM)

    stdout = io.StringIO()
    with pytest.raises(Terminated):
        serve(handle, io.StringIO(_line(tmp_path) + "\n"), stdout)
    assert stdout.getvalue() == ""


# ---------------------------------------------------------------------------
# The real worker over the shipped checkpoint (pretrained models faked)
# ---------------------------------------------------------------------------


def _without_timings(document: dict) -> dict:
    trimmed = dict(document)
    trimmed.pop("timings_seconds")
    return trimmed


@needs_serving_fixtures
def test_the_worker_answers_exactly_what_infer_rank_answers(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from slotify_rank.inference.predictor import RankerPredictor

    install(monkeypatch)
    # An empty hub cache: `infer rank` then stays online (and says so) rather
    # than switching this test process's environment to offline mode.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    system_temp = tmp_path / "system-temp"
    system_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_temp))

    one_shot_work = tmp_path / "one-shot"
    one_shot_work.mkdir()
    output = tmp_path / "one-shot.json"
    assert main([
        "infer", "rank",
        "--audio", str(FIXTURE_AUDIO),
        "--checkpoint", str(SHIPPED_CHECKPOINT),
        "--work-dir", str(one_shot_work),
        "--output", str(output),
        "--quiet",
    ]) == 0
    one_shot = json.loads(output.read_text(encoding="utf-8"))
    assert list(one_shot_work.iterdir()) == [], "infer rank left scratch files behind"

    configs = PipelineConfigs.load()
    worker = RankerWorker(predictor=RankerPredictor.load(SHIPPED_CHECKPOINT), configs=configs)
    lines = []
    for index in range(2):
        work = tmp_path / f"work-{index}"
        work.mkdir()
        lines.append(json.dumps({
            "id": index,
            "audio_path": str(FIXTURE_AUDIO),
            "work_dir": str(work),
            "episode_id": "upload",
            "skip_transcription": False,
        }))
    with resident.resident_models():
        preload_models(configs)
        code, responses = _serve(lines, worker.handle)

    assert code == 0
    assert [r["ok"] for r in responses] == [True, True]
    for response in responses:
        assert _without_timings(response["payload"]) == _without_timings(one_shot)
        # Loaded once at startup, not by this request.
        assert response["payload"]["timings_seconds"]["model_load"] == 0.0
    assert one_shot["timings_seconds"]["model_load"] > 0.0
    for index in range(2):
        assert list((tmp_path / f"work-{index}").iterdir()) == []
    assert list(system_temp.iterdir()) == [], "a request wrote to the system temp dir"


@needs_serving_fixtures
def test_skip_transcription_masks_text_and_says_so(tmp_path: Path, monkeypatch) -> None:
    from slotify_rank.inference.predictor import RankerPredictor

    install(monkeypatch)
    worker = RankerWorker(
        predictor=RankerPredictor.load(SHIPPED_CHECKPOINT), configs=PipelineConfigs.load()
    )
    request = WorkerRequest(
        id=1, audio_path=FIXTURE_AUDIO, work_dir=tmp_path, skip_transcription=True
    )
    document = worker.handle(request)
    assert all(not candidate["text_available"] for candidate in document["ranked"])
    assert any("Transcription was skipped" in warning for warning in document["warnings"])
    assert LOADS.get("transcriber", 0) == 0


@needs_serving_fixtures
def test_a_missing_audio_file_is_a_request_failure_not_a_crash(tmp_path: Path, monkeypatch) -> None:
    from slotify_rank.inference.predictor import RankerPredictor

    install(monkeypatch)
    worker = RankerWorker(
        predictor=RankerPredictor.load(SHIPPED_CHECKPOINT), configs=PipelineConfigs.load()
    )
    code, responses = _serve(
        [
            json.dumps({"id": "gone", "audio_path": str(tmp_path / "nope.mp3"), "work_dir": str(tmp_path)}),
            json.dumps({"id": "real", "audio_path": str(FIXTURE_AUDIO), "work_dir": str(tmp_path)}),
        ],
        worker.handle,
    )
    assert code == 0
    assert responses[0]["ok"] is False and "nope.mp3" in responses[0]["error"]
    assert responses[1]["ok"] is True
    assert list(tmp_path.iterdir()) == []


def test_infer_rank_refuses_both_scratch_options(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        main([
            "infer", "rank", "--audio", "a.mp3", "--checkpoint", "c.pt",
            "--work-dir", str(tmp_path), "--workspace", str(tmp_path / "w"),
        ])
    assert caught.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The real `infer serve` process
# ---------------------------------------------------------------------------


def _fake_hub(tmp_path: Path, present: bool) -> Path:
    """A hub cache holding the committed configs' pinned snapshots, or nothing."""
    from slotify_rank.inference.hub import pinned_models, snapshot_dir

    root = tmp_path / "hub"
    root.mkdir()
    if present:
        env = {"HF_HUB_CACHE": str(root)}
        for model in pinned_models(PipelineConfigs.load()):
            directory = snapshot_dir(model, env)
            directory.mkdir(parents=True)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "model.safetensors").write_bytes(b"w")
    return root


def _spawn_worker(tmp_path: Path, hub_present: bool = True, **extra_env: str) -> subprocess.Popen:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HF_", "TRANSFORMERS_", "SENTENCE_TRANSFORMERS_"))
    }
    env.update(
        HF_HUB_CACHE=str(_fake_hub(tmp_path, hub_present)),
        PYTHONPATH=os.pathsep.join([str(ML_DIR / "src"), str(ML_DIR)]),
        **extra_env,
    )
    return subprocess.Popen(
        [
            sys.executable, "-c", BOOTSTRAP,
            "infer", "serve", "--checkpoint", str(SHIPPED_CHECKPOINT),
        ],
        cwd=ML_DIR,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )


def _request(work_dir: Path, request_id: str) -> str:
    return json.dumps(
        {"id": request_id, "audio_path": str(FIXTURE_AUDIO), "work_dir": str(work_dir)}
    ) + "\n"


@needs_serving_fixtures
def test_the_serve_process_keeps_stdout_for_the_protocol(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    process = _spawn_worker(tmp_path, SLOTIFY_FAKE_STDOUT_NOISE="1")
    stdout, stderr = process.communicate(
        _request(work, "a") + _request(work, "b"), timeout=240
    )
    assert process.returncode == 0, stderr
    responses = [json.loads(line) for line in stdout.splitlines()]
    assert [(r["id"], r["ok"]) for r in responses] == [("a", True), ("b", True)], stdout
    # The fake transcriber's print() went to stderr, not into the protocol.
    assert "fake transcriber chatter on stdout" in stderr
    assert "HF_HUB_OFFLINE=1" in stderr
    assert "worker: ready" in stderr
    assert list(work.iterdir()) == []


@needs_serving_fixtures
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_mid_request_exits_cleanly_and_removes_the_scratch(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    marker = tmp_path / "transcribing"
    process = _spawn_worker(tmp_path, SLOTIFY_FAKE_HANG_MARKER=str(marker))
    assert process.stdin is not None
    process.stdin.write(_request(work, "slow"))
    process.stdin.flush()
    deadline = time.monotonic() + 180
    while not marker.exists():
        assert process.poll() is None, process.stderr.read() if process.stderr else ""
        assert time.monotonic() < deadline, "the worker never reached transcription"
        time.sleep(0.1)
    assert any(work.iterdir()), "the request should be using its work dir"

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    assert stdout == ""  # no response for an abandoned request
    assert "worker: stopping" in stderr
    assert list(work.iterdir()) == []


@needs_serving_fixtures
def test_the_worker_refuses_to_start_without_its_pinned_snapshots(tmp_path: Path) -> None:
    process = _spawn_worker(tmp_path, hub_present=False)
    stdout, stderr = process.communicate("", timeout=120)
    assert process.returncode == 1
    assert stdout == ""
    last = [line for line in stderr.splitlines() if line.strip()][-1]
    assert last.startswith("error: Pinned model snapshot(s)")
    assert WARM_COMMAND in last
