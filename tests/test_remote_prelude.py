"""File-based prelude for remote launches.

A multi-KB prelude inlined into the remote launch command overruns the
remote shell's command-length limit ("command too long").  On remote the
prelude is written to a file over SSH stdin and referenced as
``"$(cat <file>)"``, keeping the command line short while the agent's own
shell expands it at launch.
"""
from tests.test_remote import _build_remote_manager

_SENTINEL = "__SUCODER_PRELUDE_FROM_FILE__"


def test_externalize_prelude_writes_file_and_substitutes(tmp_path):
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")

    recorded = {}

    def spy(args, *, input=None, **kwargs):
        recorded["args"] = list(args)
        recorded["input"] = input

    manager.executor.run_agent = spy

    prelude = "SYSTEM PROMPT\n" + ("x" * 5000)  # large, multi-line
    cmd = f"claude --system-prompt {_SENTINEL} --dangerously-skip-permissions; exec bash -l"
    out = manager._externalize_prelude(cmd, _SENTINEL, prelude, ctx)

    # The prelude is gone from the command line, replaced by a cat ref.
    assert _SENTINEL not in out
    assert "SYSTEM PROMPT" not in out
    assert "xxxxx" not in out
    assert '"$(cat "$HOME/.cache/sucoder/prelude-rproj.txt")"' in out

    # The prelude went over stdin, never into argv.
    assert recorded["input"] == prelude
    assert recorded["args"][0] == "sh"
    assert any("cat >" in part for part in recorded["args"])
    assert all(prelude not in part for part in recorded["args"])
