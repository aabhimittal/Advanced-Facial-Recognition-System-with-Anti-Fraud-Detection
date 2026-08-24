"""CLI smoke tests — the demo commands are documentation that has to keep working."""

import pytest

from faceguard.cli import main


@pytest.mark.parametrize("argv", [["policy"], ["attacks", "--frames", "60"], ["stream", "--frames", "90"]])
def test_commands_exit_cleanly(argv, capsys):
    assert main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_attacks_command_reports_every_class(capsys):
    main(["attacks", "--frames", "90"])
    out = capsys.readouterr().out
    for label in ("live face", "injected stream", "3-D silicone mask", "screen replay"):
        assert label in out
    assert "MISS" not in out


def test_stream_command_leaves_an_intact_audit_chain(capsys):
    main(["stream", "--sample", "spoof", "--frames", "90"])
    out = capsys.readouterr().out
    assert "audit chain intact: True" in out
    assert "FRAUD" in out


def test_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        main(["not-a-command"])
