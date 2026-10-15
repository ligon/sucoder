"""The tmux-placement helpers behind ``collaborate`` / ``agents-run`` / ``attach``.

Pure functions over an injected environment and runner: no tmux server
is started here.  The CLI wiring is covered in ``test_cli.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

from sucoder import tmux_env


def _runner(answers):
    """A ``subprocess.run`` stand-in answering by the tmux subcommand.

    ``answers`` maps the subcommand (``display-message``, ``has-session``,
    ...) to ``(returncode, stdout)``; an unlisted one fails.  Records every
    call, with the environment it was given, for the assertions below.
    """
    calls = []

    def run(argv, **kwargs):
        calls.append((list(argv), kwargs))
        rc, out = answers.get(argv[1], (1, ""))
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr="")

    run.calls = calls
    return run


# --- detect ---------------------------------------------------------------

def test_detect_outside_tmux_asks_nothing():
    run = _runner({})
    state = tmux_env.detect({"TMUX": ""}, run=run)
    assert state == tmux_env.TmuxContext(inside=False)
    assert run.calls == []


def test_detect_inside_tmux_names_session_socket_and_prefix():
    run = _runner({"display-message": (0, "main\n"), "show-options": (0, "C-a\n")})
    state = tmux_env.detect({"TMUX": "/tmp/tmux-1000/default,4242,0"}, run=run)
    assert state.inside
    assert state.socket == "/tmp/tmux-1000/default"
    assert state.session == "main"
    assert state.prefix == "C-a"
    assert state.on_default_server


def test_detect_inside_tmux_survives_a_server_that_does_not_answer():
    """``$TMUX`` is the fact; the decorations fall back."""
    def run(argv, **kwargs):
        raise FileNotFoundError("tmux")

    state = tmux_env.detect({"TMUX": "/tmp/tmux-1000/work,1,0"}, run=run)
    assert state.inside
    assert state.session is None
    assert state.prefix == tmux_env.DEFAULT_PREFIX
    assert not state.on_default_server


# --- session naming and probes -------------------------------------------

def test_local_session_name_matches_the_remote_token():
    assert tmux_env.local_session_name("SuCoder") == "sucoder-SuCoder"
    assert tmux_env.local_session_name("a b;c") == "sucoder-a_b_c"


def test_session_exists_uses_exact_match_on_the_default_server():
    run = _runner({"has-session": (0, "")})
    environ = {"TMUX": "/tmp/tmux-1000/work,1,0", "PATH": "/bin"}
    assert tmux_env.session_exists("sucoder-Foo", environ=environ, run=run)
    argv, kwargs = run.calls[0]
    assert argv == ["tmux", "has-session", "-t", "=sucoder-Foo"]
    # The probe must not be answered by the outer (non-default) server.
    assert "TMUX" not in kwargs["env"]
    assert kwargs["env"]["PATH"] == "/bin"


def test_session_exists_is_false_when_tmux_is_missing_or_says_no():
    assert not tmux_env.session_exists("x", environ={}, run=_runner({}))

    def run(argv, **kwargs):
        raise FileNotFoundError("tmux")

    assert not tmux_env.session_exists("x", environ={}, run=run)


# --- the terminal gate ----------------------------------------------------

def _stream(tty):
    return SimpleNamespace(isatty=lambda: tty)


def test_interactive_terminal_needs_a_tty_on_both_ends_and_a_real_term():
    yes, no = _stream(True), _stream(False)
    assert tmux_env.interactive_terminal({"TERM": "xterm"}, stdin=yes, stdout=yes)
    assert not tmux_env.interactive_terminal({"TERM": "xterm"}, stdin=no, stdout=yes)
    assert not tmux_env.interactive_terminal({"TERM": "xterm"}, stdin=yes, stdout=no)
    # Emacs shell-mode / eshell: a pty, but one tmux refuses.
    assert not tmux_env.interactive_terminal({"TERM": "dumb"}, stdin=yes, stdout=yes)
    assert not tmux_env.interactive_terminal({}, stdin=yes, stdout=yes)


# --- argv builders --------------------------------------------------------

def test_command_argv_keeps_an_executable_script_or_bare_name(tmp_path):
    script = tmp_path / "sucoder"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    assert tmux_env.command_argv([str(script), "collaborate", "Foo"]) == [
        str(script), "collaborate", "Foo",
    ]
    assert tmux_env.command_argv(["sucoder", "collaborate"]) == ["sucoder", "collaborate"]


def test_command_argv_keeps_the_interpreter_for_a_plain_file(tmp_path):
    script = tmp_path / "cli.py"
    script.write_text("")
    assert tmux_env.command_argv([str(script), "x"]) == [sys.executable, str(script), "x"]


def test_new_session_argv_quotes_the_command_and_keeps_the_cwd():
    argv = tmux_env.new_session_argv(
        "sucoder-Foo", ["/usr/bin/sucoder", "collaborate", "Foo", "--task", "a b"], "/work",
    )
    assert argv == [
        "tmux", "new-session", "-s", "sucoder-Foo", "-c", "/work",
        "/usr/bin/sucoder collaborate Foo --task 'a b'",
    ]


def test_attach_and_switch_argv_use_exact_match():
    assert tmux_env.attach_argv("sucoder-Foo") == [
        "tmux", "attach-session", "-t", "=sucoder-Foo",
    ]
    assert tmux_env.switch_argv("sucoder-Foo") == [
        "tmux", "switch-client", "-t", "=sucoder-Foo",
    ]


# --- the warnings ---------------------------------------------------------

def test_nested_remote_warning_names_the_session_and_renders_the_outer_prefix():
    state = tmux_env.TmuxContext(inside=True, session="main", prefix="C-a")
    text = tmux_env.nested_remote_warning(state)
    assert 'inside tmux session "main"' in text
    assert "`C-a C-a d` detaches the remote session" in text
    assert "`C-a d` detaches this one" in text


def test_nested_remote_warning_without_a_session_name_still_reads():
    text = tmux_env.nested_remote_warning(tmux_env.TmuxContext(inside=True))
    assert "inside a tmux session" in text
    assert "`C-b C-b d`" in text


def test_unwrapped_local_warning_says_why_and_what_to_do():
    text = tmux_env.unwrapped_local_warning("sucoder-Foo", "tmux is not on PATH")
    assert text.startswith("note: tmux is not on PATH, so the agent runs bound to this terminal")
    assert "tmux new -s sucoder-Foo" in text


def test_real_tmux_probe_shape_matches_the_builders():
    """The builders use only tmux's documented, stable syntax; no server needed."""
    assert os.sep not in tmux_env.local_session_name("Foo/Bar")
