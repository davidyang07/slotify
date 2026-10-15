"""runtime.py and the CLIs' stop/cleanup contract: SIGTERM exits 143, temp
files stay in --work-dir and are gone afterwards, cloned voices are released.

The subprocess tests run each CLI the way the Node API does (``cwd`` =
``backend/``, a fresh interpreter) with a driver that stubs only what needs a
network or a model, against the loopback ``fake_api``. No network.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from ad_inserter import runtime

from conftest import BACKEND_DIR, RecordedRequest


def _settle() -> None:
    """Give a pending signal handler a few bytecode boundaries to run."""
    for _ in range(20):
        time.sleep(0.005)


# --- runtime, in process ----------------------------------------------------


def test_sigterm_in_a_session_raises_system_exit_143_and_cleans_up(
    tmp_path: Path,
) -> None:
    previous = signal.getsignal(signal.SIGTERM)
    previous_tempdir = tempfile.tempdir

    with pytest.raises(SystemExit) as stopped:
        with runtime.cli_session(tmp_path) as private:
            assert signal.getsignal(signal.SIGTERM) is runtime._handle_signal
            (private / "partial.wav").write_bytes(b"x")
            signal.raise_signal(signal.SIGTERM)
            _settle()
            pytest.fail("SIGTERM did not stop the session")

    assert stopped.value.code == 143
    assert list(tmp_path.iterdir()) == []
    assert signal.getsignal(signal.SIGTERM) is previous
    assert tempfile.tempdir == previous_tempdir


def test_sigint_exits_130() -> None:
    with pytest.raises(SystemExit) as stopped:
        with runtime.cli_session():
            assert signal.getsignal(signal.SIGINT) is runtime._handle_signal
            signal.raise_signal(signal.SIGINT)
            _settle()

    assert stopped.value.code == 130


def test_a_shielded_step_finishes_before_the_exit() -> None:
    steps: List[str] = []

    with pytest.raises(SystemExit) as stopped:
        with runtime.cli_session():
            with runtime.shielded():
                signal.raise_signal(signal.SIGTERM)
                _settle()
                assert runtime.termination_requested()
                steps.append("finished the shielded step")
            steps.append("ran past the shield")

    assert stopped.value.code == 143
    assert steps == ["finished the shielded step"]
    assert not runtime.termination_requested()  # reset once the session is over


def test_a_second_signal_does_not_interrupt_cleanup() -> None:
    cleaned: List[bool] = []

    with pytest.raises(SystemExit) as stopped:
        with runtime.cli_session():
            try:
                signal.raise_signal(signal.SIGTERM)
                _settle()
            finally:
                signal.raise_signal(signal.SIGTERM)
                _settle()
                cleaned.append(True)

    assert stopped.value.code == 143
    assert cleaned == [True]


def test_temp_files_go_in_the_work_dir_and_are_removed(tmp_path: Path) -> None:
    previous_tmpdir_env = os.environ.get("TMPDIR")

    with runtime.cli_session(tmp_path) as private:
        assert private.parent == tmp_path
        assert Path(tempfile.gettempdir()) == private
        assert os.environ["TMPDIR"] == str(private)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            leaked = Path(handle.name)
        assert leaked.parent == private

    assert not leaked.exists()
    assert list(tmp_path.iterdir()) == []
    assert os.environ.get("TMPDIR") == previous_tmpdir_env


def test_without_a_work_dir_the_private_dir_is_in_the_system_temp_dir() -> None:
    system_tmp = Path(tempfile.gettempdir())

    with runtime.cli_session() as private:
        assert private.parent == system_tmp

    assert not private.exists()


def test_work_dir_must_be_an_existing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser()
    runtime.add_work_dir_argument(parser)

    assert parser.parse_args([f"--work-dir={tmp_path}"]).work_dir == tmp_path
    assert parser.parse_args([]).work_dir is None
    with pytest.raises(SystemExit) as refused:
        parser.parse_args([f"--work-dir={tmp_path / 'missing'}"])
    assert refused.value.code == 2
    assert "not an existing directory" in capsys.readouterr().err


# --- the CLIs as subprocesses -----------------------------------------------


def _wait_for(condition: Callable[[], bool], proc: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if condition():
            return
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    proc.kill()
    proc.wait()
    pytest.fail(f"child never reached the blocking stub:\n{log.read_text(errors='ignore')}")


def _spawn(
    tmp_path: Path, driver: str, args: List[str], env: Dict[str, str]
) -> Tuple[subprocess.Popen, Path, Path]:
    """Start ``driver`` like Node starts a CLI; returns (process, system tmp, log).

    The child leads its own process group, as Node spawns it, so ``_stop`` can
    signal the whole tree the way Node does.
    """
    system_tmp = tmp_path / "system-tmp"
    system_tmp.mkdir()
    log = tmp_path / "child.log"
    child_env = dict(os.environ)  # the autouse fixture already removed credentials
    child_env.update(env)
    child_env["TMPDIR"] = str(system_tmp)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log.open("wb") as log_handle:
        proc = subprocess.Popen(
            [sys.executable, "-c", driver, *args],
            cwd=BACKEND_DIR,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc, system_tmp, log


def _stop(proc: subprocess.Popen, log: Path, group: bool = False) -> int:
    if group:
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.send_signal(signal.SIGTERM)
    try:
        return proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail(f"child ignored SIGTERM:\n{log.read_text(errors='ignore')}")


def _tone_with_gaps(path: Path, gaps_at_s: List[int], duration_s: int = 20) -> Path:
    audio = Sine(300).to_audio_segment(duration=duration_s * 1000, volume=-12)
    for start_s in gaps_at_s:
        start = start_s * 1000
        audio = audio[:start] + AudioSegment.silent(1_000) + audio[start + 1_000 :]
    audio.export(path, format="wav")
    return path


ANALYZE_DRIVER = """
import os, sys, time, types

class BlockingModel:
    def transcribe(self, path, fp16=False):
        with open(os.environ["TEST_MARKER"], "w") as handle:
            handle.write(path)
        while True:
            time.sleep(0.05)

whisper = types.ModuleType("whisper")
whisper.load_model = lambda name: BlockingModel()
sys.modules["whisper"] = whisper

from ad_inserter import analyze_cli
sys.argv = ["analyze_cli", *sys.argv[1:]]
analyze_cli.run()
"""


@pytest.mark.ffmpeg
@pytest.mark.parametrize("with_work_dir", [True, False])
def test_sigterm_stops_analyze_cli_and_leaves_no_temp_files(
    tmp_path: Path, with_work_dir: bool
) -> None:
    audio = _tone_with_gaps(tmp_path / "episode.wav", [5, 12])
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    marker = tmp_path / "transcribing"
    args = [f"--audio={audio}", "--mode=podcast"]
    if with_work_dir:
        args.append(f"--work-dir={work_dir}")

    proc, system_tmp, log = _spawn(
        tmp_path, ANALYZE_DRIVER, args, {"TEST_MARKER": str(marker)}
    )
    _wait_for(marker.exists, proc, log)
    snippet = Path(marker.read_text())
    assert snippet.exists()
    # The snippet WAV sits in the run's private dir, inside --work-dir when given.
    assert snippet.parent.parent == (work_dir if with_work_dir else system_tmp)

    assert _stop(proc, log, group=True) == 143
    assert not snippet.exists()
    assert list(work_dir.iterdir()) == []
    assert list(system_tmp.iterdir()) == []


FAKE_FFMPEG = """#!/bin/sh
# Stands in for ffmpeg: writes part of its output file, then hangs.
for last; do :; done
printf partial > "$last"
echo $$ > "$TEST_MARKER"
exec sleep 60
"""

PLAIN_ANALYZE_DRIVER = """
import sys
from ad_inserter import analyze_cli
sys.argv = ["analyze_cli", *sys.argv[1:]]
analyze_cli.run()
"""


@pytest.mark.parametrize("group", [False, True], ids=["process", "process-group"])
def test_sigterm_during_an_ffmpeg_decode_stops_it_and_removes_its_output(
    tmp_path: Path, group: bool
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "ffmpeg"
    fake.write_text(FAKE_FFMPEG)
    fake.chmod(0o755)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"not decoded by the stub")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    marker = tmp_path / "decoding"
    env = {"TEST_MARKER": str(marker), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

    proc, system_tmp, log = _spawn(
        tmp_path, PLAIN_ANALYZE_DRIVER, [f"--audio={audio}", f"--work-dir={work_dir}"], env
    )
    _wait_for(lambda: marker.exists() and marker.read_text().strip() != "", proc, log)
    ffmpeg_pid = int(marker.read_text())
    assert [p.suffix for p in work_dir.glob("*/*")] == [".s16le"]

    assert _stop(proc, log, group=group) == 143
    assert list(work_dir.iterdir()) == []
    assert list(system_tmp.iterdir()) == []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(ffmpeg_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the ffmpeg child outlived the stopped CLI")


CLI_DRIVER = """
import sys
from pydub.generators import Sine
from ad_inserter import analysis, cli, tts

analysis.whisper_available = lambda: False
tts.synthesize_audio = lambda request: Sine(440).to_audio_segment(duration=1500, volume=-10)
sys.argv = ["cli", *sys.argv[1:]]
cli.run()
"""


@pytest.mark.ffmpeg
def test_sigterm_stops_cli_during_the_merge_and_leaves_no_temp_files(
    tmp_path: Path, fake_api
) -> None:
    audio = _tone_with_gaps(tmp_path / "episode.wav", [5, 12, 16])
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    out = tmp_path / "out.mp3"
    merging, release = threading.Event(), threading.Event()

    def responder(_request: RecordedRequest) -> Tuple[int, bytes]:
        merging.set()
        release.wait(30)
        return 500, b"released"

    fake_api.respond_with(responder)
    args = [
        f"--main={audio}",
        "--voice-id=voice-1",
        "--product-name=Sparrow Notes",
        "--product-desc=A calmer note-taking app",
        f"--out={out}",
        "--llm-provider=none",
        f"--merge-url={fake_api.url}/api/merge",
        f"--work-dir={work_dir}",
    ]
    proc, system_tmp, log = _spawn(tmp_path, CLI_DRIVER, args, {})
    try:
        _wait_for(merging.is_set, proc, log)
        # The promo MP3 being uploaded is a temp file in the work dir.
        assert [p.suffix for p in work_dir.glob("*/*")] == [".mp3"]

        assert _stop(proc, log) == 143
    finally:
        release.set()
    assert list(work_dir.iterdir()) == []
    assert list(system_tmp.iterdir()) == []
    assert not out.exists()


INSERT_DRIVER = """
import os, sys
from ad_inserter import analysis, insert_ad

analysis.diarize_speakers = lambda _path: [
    analysis.SpeakerSegment(speaker="A", start_ms=0, end_ms=3000),
    analysis.SpeakerSegment(speaker="B", start_ms=3900, end_ms=6900),
]
insert_ad.ELEVENLABS_VOICES_URL = os.environ["TEST_VOICES_URL"]
sys.argv = ["insert_ad", *sys.argv[1:]]
insert_ad.run()
"""


@pytest.mark.ffmpeg
def test_sigterm_stops_insert_ad_and_it_still_cleans_up_its_cloned_voices(
    tmp_path: Path, fake_api
) -> None:
    audio = _tone_with_gaps(tmp_path / "conversation.wav", [3, 7, 11], duration_s=16)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    out = tmp_path / "out.mp3"
    synthesizing, release = threading.Event(), threading.Event()

    def responder(request: RecordedRequest) -> Tuple[int, bytes]:
        if request.path == "/api/clone":
            speaker = re.search(rb"Speaker (\w) Clone", request.body).group(1).decode()
            return 200, json.dumps({"voiceId": f"clone-{speaker}"}).encode()
        if request.path == "/api/tts":
            synthesizing.set()
            release.wait(30)
            return 500, b"released"
        if request.path == "/api/clone/clone-B/release":
            return 404, b"unknown voice"  # not in the registry: deleted directly
        return 200, b""

    fake_api.respond_with(responder)
    args = [
        f"--input={audio}",
        "--product-name=Sparrow Notes",
        "--product-blurb=A calmer note-taking app",
        "--ad-style=casual",
        "--ad-mode=DUO",
        f"--out={out}",
        "--llm-provider=none",
        "--clone-voices",
        f"--clone-url={fake_api.url}/api/clone",
        f"--tts-url={fake_api.url}/api/tts",
        f"--work-dir={work_dir}",
    ]
    env = {
        "ELEVENLABS_API_KEY": "eleven-key",
        "SLOTIFY_INTERNAL_TOKEN": "internal-token",
        "TEST_VOICES_URL": f"{fake_api.url}/v1/voices",
    }
    proc, system_tmp, log = _spawn(tmp_path, INSERT_DRIVER, args, env)
    try:
        _wait_for(synthesizing.is_set, proc, log)
        assert _stop(proc, log, group=True) == 143
    finally:
        release.set()

    cleanup = [request for request in fake_api.requests if request.path != "/api/clone"]
    assert [(request.method, request.path) for request in cleanup] == [
        ("POST", "/api/tts"),
        ("POST", "/api/clone/clone-A/release"),
        ("POST", "/api/clone/clone-B/release"),
        ("DELETE", "/v1/voices/clone-B"),
    ]
    for request in cleanup[1:3]:
        assert request.headers["authorization"] == "Bearer internal-token"
    assert cleanup[3].headers["xi-api-key"] == "eleven-key"
    assert "authorization" not in cleanup[3].headers
    assert list(work_dir.iterdir()) == []
    assert list(system_tmp.iterdir()) == []
    assert not out.exists()
