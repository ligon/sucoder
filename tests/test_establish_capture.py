"""Integration tests for ``SshControl.establish`` stderr capture.

These exercise the *real* ``subprocess.run`` against a fake ``ssh`` placed
on ``PATH`` (the other establish tests mock ``run`` and so cannot catch
the two hazards covered here):

1. On failure, the ssh stderr must be captured onto ``TunnelError.stderr``
   so callers can classify a transient ``kex_exchange_identification``
   closure and retry it.
2. ``establish`` uses ``ssh -fN``: the ControlMaster forks and holds its
   stderr open for the life of the connection.  Capturing via
   ``subprocess.PIPE`` would block on EOF and hang on *success*; capture
   must therefore use a regular file so ``run`` returns as soon as the
   foreground process exits.
"""
import logging
import os
import threading

import pytest

from sucoder.tunnel import SshControl, TunnelError

_LOG = logging.getLogger("test.establish")


def _install_fake_ssh(tmp_path, monkeypatch, script: str):
    """Write an executable fake ``ssh`` and put it first on ``PATH``."""
    fake = tmp_path / "ssh"
    fake.write_text(script)
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")


def _make_control(tmp_path, monkeypatch):
    socket_file = tmp_path / "gw.sock"
    monkeypatch.setattr(
        SshControl, "socket_path", property(lambda self: socket_file),
    )
    control = SshControl(gateway="gw")
    monkeypatch.setattr(control, "is_active", lambda: False)
    return control


def test_establish_captures_stderr_on_failure(tmp_path, monkeypatch):
    _install_fake_ssh(
        tmp_path, monkeypatch,
        "#!/bin/sh\n"
        "echo 'kex_exchange_identification: Connection closed by remote host' >&2\n"
        "exit 255\n",
    )
    control = _make_control(tmp_path, monkeypatch)

    with pytest.raises(TunnelError) as excinfo:
        control.establish(_LOG)

    # The transient reason must be on the exception for classification.
    assert "kex_exchange_identification" in excinfo.value.stderr


def test_establish_does_not_hang_when_master_backgrounds(tmp_path, monkeypatch):
    # Foreground writes a negotiation line then exits 0, but a "master"
    # child keeps stderr open (mimicking ``ssh -fN``).  With a PIPE this
    # would block on EOF; with a regular file it returns immediately.
    _install_fake_ssh(
        tmp_path, monkeypatch,
        "#!/bin/sh\n"
        "echo 'debug1: pretend negotiation' >&2\n"
        "( sleep 30 ) &\n"          # child inherits stderr, stays alive
        "exit 0\n",                 # foreground exits at once
    )
    control = _make_control(tmp_path, monkeypatch)

    done = threading.Event()

    def _run():
        control.establish(_LOG)
        done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # On the correct (temp-file) implementation this returns in well under
    # a second; a PIPE-based capture would block ~30s on the held stderr.
    assert done.wait(timeout=5), "establish() hung capturing a backgrounded master's stderr"


def test_sshcontrol_rejects_a_none_host():
    """A None host must fail at construction, naming the real problem.

    ``gateway`` is typed non-Optional but is fed from session state that can
    be None (``session.compute_node`` when a SLURM job was recorded before
    its node resolved).  Unchecked, the None rode all the way into
    ``subprocess.run(["ssh", ..., None])`` and surfaced as ``TypeError:
    expected str, bytes or os.PathLike object, not NoneType`` from inside
    Popen -- a traceback that names neither the host nor the caller, and
    which reads as a Python bug rather than unresolved session state.
    """
    from sucoder.tunnel import SshControl, TunnelError

    with pytest.raises(TunnelError) as excinfo:
        SshControl(gateway=None)
    assert "no host" in str(excinfo.value).lower()

    with pytest.raises(TunnelError):
        SshControl(gateway="")
