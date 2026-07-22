"""CLI smoke tests: exit codes, machine-readable output, and Windows paths.

These run entirely offline and use ``tmp_path`` so they exercise absolute paths
containing spaces on Windows the same way the developer's OneDrive checkout does.
"""

from __future__ import annotations

import json

import pytest

from slotify_rank.cli import main

EPISODE = {
    "episode_id": "ep-cli",
    "duration_seconds": 600.0,
    "mode": "podcast",
    "count": 3,
    "silence_candidates": [
        {"ms": 60000, "silence_ms": 1800, "snippet": "So that was the whole story."},
        {"ms": 180000, "silence_ms": 300, "snippet": "That's exactly right."},
        {"ms": 300000, "silence_ms": 2400, "snippet": "We'll come back to that."},
    ],
    "transcript_candidates": [],
}


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_version_command(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "canonical_baseline: heuristic_offline_v1" in out


def test_config_show_command(capsys):
    assert main(["config", "show"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["canonical_profile"] == "heuristic_offline_v1"
    assert payload["profile"]["scoring"]["base_score"] == 0.4


def test_config_show_rejects_unknown_profile(capsys):
    assert main(["config", "show", "--profile", "nope"]) == 1
    assert "Unknown heuristic profile" in capsys.readouterr().err


def test_heuristic_rank_writes_machine_readable_output(tmp_path, capsys):
    source = _write(tmp_path / "episodes.json", {"episodes": [EPISODE]})
    destination = tmp_path / "out" / "rankings.json"

    assert main(["heuristic", "rank", "--input", str(source), "--output", str(destination)]) == 0

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["baseline_version"] == "heuristic_offline_v1"
    assert len(payload["episodes"]) == 1
    episode = payload["episodes"][0]
    assert episode["episode_id"] == "ep-cli"
    assert len(episode["ranked_candidate_ids"]) == 3
    assert len(episode["points"]) == 3
    assert "Ranked 1 episode(s)" in capsys.readouterr().out


def test_heuristic_rank_accepts_a_bare_episode_object(tmp_path):
    source = _write(tmp_path / "episode.json", EPISODE)
    destination = tmp_path / "rankings.json"
    assert main(["heuristic", "rank", "--input", str(source), "--output", str(destination)]) == 0


def test_heuristic_rank_reports_missing_input(tmp_path, capsys):
    assert (
        main(
            [
                "heuristic",
                "rank",
                "--input",
                str(tmp_path / "absent.json"),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 1
    )
    assert "Input file not found" in capsys.readouterr().err


def test_heuristic_rank_reports_malformed_json(tmp_path, capsys):
    source = tmp_path / "bad.json"
    source.write_text("{not json", encoding="utf-8")
    assert (
        main(["heuristic", "rank", "--input", str(source), "--output", str(tmp_path / "o.json")])
        == 1
    )
    assert "not valid JSON" in capsys.readouterr().err


def test_heuristic_rank_reports_empty_episode_list(tmp_path, capsys):
    source = _write(tmp_path / "empty.json", {"episodes": []})
    assert (
        main(["heuristic", "rank", "--input", str(source), "--output", str(tmp_path / "o.json")])
        == 1
    )
    assert "contains no episodes" in capsys.readouterr().err


def test_evaluate_end_to_end(tmp_path, capsys):
    source = _write(tmp_path / "episodes.json", {"episodes": [EPISODE]})
    rankings = tmp_path / "rankings.json"
    assert main(["heuristic", "rank", "--input", str(source), "--output", str(rankings)]) == 0

    ranked = json.loads(rankings.read_text(encoding="utf-8"))["episodes"][0][
        "ranked_candidate_ids"
    ]
    labels = _write(
        tmp_path / "labels.json",
        {
            "episodes": [
                {
                    "episode_id": "ep-cli",
                    "relevance": {
                        ranked[0]: 5.0,
                        ranked[1]: 4.0,
                        ranked[2]: 2.0,
                    },
                }
            ]
        },
    )
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(rankings),
                "--labels",
                str(labels),
                "--output",
                str(report),
            ]
        )
        == 0
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["n_episodes"] == 1
    assert payload["aggregate"]["ndcg_at_k"] == pytest.approx(1.0)
    assert "Episodes evaluated: 1" in capsys.readouterr().out


def test_evaluate_rejects_labels_without_relevance(tmp_path, capsys):
    source = _write(tmp_path / "episodes.json", {"episodes": [EPISODE]})
    rankings = tmp_path / "rankings.json"
    main(["heuristic", "rank", "--input", str(source), "--output", str(rankings)])
    labels = _write(tmp_path / "labels.json", {"episodes": [{"episode_id": "ep-cli"}]})
    assert main(["evaluate", "--predictions", str(rankings), "--labels", str(labels)]) == 1
    assert "missing 'relevance'" in capsys.readouterr().err


def test_unknown_command_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["nonexistent"])
    assert excinfo.value.code == 2
