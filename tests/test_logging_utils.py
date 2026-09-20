"""Progress lines and where the machinery's own log records go.

Both exist because of the same report: `sucoder sessions` and
`sucoder -T savio collaborate` "simply hang".  Neither hung --- each was
making a dozen ~8s round trips with nothing on screen, and `-v`, the one
flag a user reaches for at that moment, added nothing, because it
configured the command's logger while the connection machinery logs to
``sucoder.tunnel``, which propagated to a root with no handlers.
"""

from __future__ import annotations

import logging

from sucoder.logging_utils import (
    describe_remote, progress, set_progress, setup_logger, summarize_command,
)


# -- describe_remote: which commands cost a round trip -----------------------

def test_ssh_command_is_named_by_host_and_remote_command():
    assert describe_remote([
        "ssh", "-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/s.sock",
        "-o", "BatchMode=yes", "ligon@hpc.brc.berkeley.edu", "squeue --me",
    ]) == ("hpc.brc.berkeley.edu", "squeue --me")


def test_ssh_options_are_not_mistaken_for_the_host():
    """``-W %h:%p`` and friends take a value; the host is the first bare
    token after them, not the value itself."""
    host, _ = describe_remote([
        "ssh", "-o", "ControlMaster=auto", "-W", "ln003.brc:22",
        "hpc.brc.berkeley.edu",
    ])
    assert host == "hpc.brc.berkeley.edu"


def test_an_interactive_ssh_still_names_its_host():
    assert describe_remote(["ssh", "-t", "-o", "ForwardX11=yes", "ln002.brc"]) == (
        "ln002.brc", "(interactive session)",
    )


def test_git_over_ssh_counts_as_remote():
    host, what = describe_remote(
        ["git", "fetch", "ln002.brc:~/mirrors/SuCoder", "+main:refs/x"],
    )
    assert host == "ln002.brc"
    assert what.startswith("git fetch")


def test_local_commands_are_not_announced():
    assert describe_remote(["git", "status", "--porcelain"]) is None
    assert describe_remote(["chgrp", "-R", "coder", "/srv/repo"]) is None
    assert describe_remote([]) is None


def test_a_script_is_summarised_by_its_first_line():
    summary = summarize_command("tmux list-sessions -F x\nwhile read -r s; do\ndone")
    assert summary.startswith("tmux list-sessions")
    assert "\n" not in summary
    assert len(summary) <= 60


# -- the lines themselves ---------------------------------------------------

def test_progress_is_silent_until_enabled(capsys):
    set_progress(False)
    progress("ln002.brc", "squeue --me")
    assert capsys.readouterr().err == ""


def test_progress_names_the_host_on_stderr(capsys):
    set_progress(True)
    try:
        progress("ln002.brc", "squeue --me")
    finally:
        set_progress(False)
    captured = capsys.readouterr()
    assert "ln002.brc" in captured.err and "squeue" in captured.err
    # stdout is a report or a value somebody may be piping; keep it clean.
    assert captured.out == ""


# -- -v must reach the connection machinery ---------------------------------

def test_setup_logger_wires_the_tunnel_logger(tmp_path):
    """``sucoder.tunnel`` explains *why* a connection was declared dead or a
    probe timed out.  Before, those records went nowhere at all."""
    tunnel_log = logging.getLogger("sucoder.tunnel")
    for handler in list(tunnel_log.handlers):
        tunnel_log.removeHandler(handler)

    setup_logger("sucoder.test-wiring", tmp_path, verbose=True)
    assert tunnel_log.handlers, "tunnel records still have nowhere to go"

    tunnel_log.debug("probe timed out")
    for handler in tunnel_log.handlers:
        handler.flush()
    assert "probe timed out" in (tmp_path / "sucoder.test-wiring.log").read_text()


def test_setup_logger_does_not_stack_handlers(tmp_path):
    """It runs once per command, and in tests many times over; a handler
    added twice would print every line twice."""
    setup_logger("sucoder.test-wiring", tmp_path, verbose=False)
    first = len(logging.getLogger("sucoder.tunnel").handlers)
    setup_logger("sucoder.test-wiring", tmp_path, verbose=False)
    assert len(logging.getLogger("sucoder.tunnel").handlers) == first
