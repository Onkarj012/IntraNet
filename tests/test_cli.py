from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from intradaynet import cli


def make_recording_console() -> Console:
    return Console(record=True, force_terminal=False, width=120)


def test_cli_home_lists_key_commands(monkeypatch):
    test_console = make_recording_console()
    monkeypatch.setattr(cli, "console", test_console)

    exit_code = cli.main([])

    output = test_console.export_text()
    assert exit_code == 0
    assert "IntradayNet CLI" in output
    assert "picks" in output
    assert "health" in output
    assert "uv run intradaynet" in output


def test_cli_dispatches_workflow_script(monkeypatch):
    test_console = make_recording_console()
    monkeypatch.setattr(cli, "console", test_console)

    captured: dict[str, object] = {}

    def fake_run(command, cwd, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["health", "--model", "runs/lgbm_v2", "--horizon", "H60"])

    assert exit_code == 0
    assert captured["cwd"] == cli.PROJECT_ROOT
    assert captured["check"] is False
    assert captured["command"] == [
        cli.sys.executable,
        str(cli.SCRIPTS_DIR / "daily_health_check.py"),
        "--model",
        "runs/lgbm_v2",
        "--horizon",
        "H60",
    ]


def test_cli_picks_dispatches_new_recommender(monkeypatch):
    test_console = make_recording_console()
    monkeypatch.setattr(cli, "console", test_console)

    captured: dict[str, object] = {}

    def fake_run(command, cwd, check):
        captured["command"] = command
        captured["cwd"] = cwd
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["picks", "--long-count", "3", "--short-count", "2"])

    assert exit_code == 0
    assert captured["cwd"] == cli.PROJECT_ROOT
    assert captured["check"] is False
    assert captured["command"] == [
        cli.sys.executable,
        str(cli.SCRIPTS_DIR / "recommend_intraday.py"),
        "--long-count",
        "3",
        "--short-count",
        "2",
    ]


def test_cli_unknown_command_returns_error(monkeypatch):
    test_console = make_recording_console()
    monkeypatch.setattr(cli, "console", test_console)

    exit_code = cli.main(["nope"])

    output = test_console.export_text()
    assert exit_code == 1
    assert "Unknown command:" in output
    assert "Available Commands" in output
