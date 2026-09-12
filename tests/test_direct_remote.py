"""Direct SSH targets must never require cluster routing or discovery."""
import json
import os
import logging
import shlex
from dataclasses import replace

import pytest
import click

from sucoder import cli, session, tunnel
from sucoder.config import Config, ConfigError, _parse_remote_config, load_config
from sucoder.executor import RemoteExecutor
from tests.test_cli import _confined_mirror_settings
from tests.test_remote import _build_remote_manager


def direct(**extra):
    return _parse_remote_config(dict(host="workstation-alias", remote_user="alice",
                                    ssh_options={"Port": "2222", "IdentityFile": "/keys/workstation"}, **extra))


@pytest.mark.parametrize("bad", [None, "", " ", 123, "-oops"])
def test_direct_host_validation(bad):
    with pytest.raises(ConfigError, match="remote.host"):
        _parse_remote_config({"host": bad})


@pytest.mark.parametrize("key", ["gateway", "transfer_host", "slurm", "cert_file"])
def test_direct_host_rejects_cluster_fields(key):
    with pytest.raises(ConfigError, match="cannot be combined"):
        _parse_remote_config({"host": "box", key: "unused"})


def test_load_direct_target_and_inline_remote(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("""human_user: tester
mirror_root: ./mirrors
targets:
  workstation:
    host: workstation-alias
    remote_user: alice
mirrors:
  sample:
    canonical_repo: ./canonical
    remote:
      host: other-box
""")
    cfg = load_config(path)
    assert cfg.targets["workstation"].host == "workstation-alias"
    assert cfg.targets["workstation"].slurm is None
    assert cfg.mirrors["sample"].remote.host == "other-box"


@pytest.fixture
def connected(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_session_dir", lambda: tmp_path / "sessions")
    monkeypatch.setattr(tunnel, "_control_socket_dir", lambda: tmp_path / "sockets")
    monkeypatch.setattr("sucoder.config.RemoteConfig._direct_ssh_identity", lambda self: "resolved")
    calls = []
    monkeypatch.setattr(cli, "_connect_with_retry", lambda control, *a, **kw: calls.append(control))
    monkeypatch.setattr(cli, "_ensure_slurm_node", lambda *a, **kw: pytest.fail("allocated Slurm"))
    # No hostname discovery or SSH subprocess is permitted during construction.
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("unexpected subprocess"))
    settings = replace(_confined_mirror_settings(tmp_path, confined=False), remote=direct())
    config = Config(human_user="tester", mirror_root=tmp_path / "mirrors", log_dir=tmp_path / "logs")
    return config, settings, calls


def test_direct_executor_uses_one_host_and_clears_stale_cluster_state(connected):
    cfg, settings, calls = connected
    stale = session.RemoteSession(mirror_name="sample", login_node="old-login",
                                  compute_node="old-compute", slurm_job_id=123)
    stale.save()
    executor = cli._build_executor(cfg, logging.getLogger("direct"), dry_run=False,
                                   mirror_settings=settings)
    assert len(calls) == 1
    assert calls[0].gateway == "workstation-alias"
    assert calls[0].jump_host is None
    assert "Port=2222" in calls[0].extra_options
    assert executor.gateway == ""
    assert executor.scaffolding_node == ""
    saved = session.RemoteSession.load("sample")
    assert saved.login_node == "workstation-alias"
    assert saved.compute_node is None and saved.slurm_job_id is None
    for socket in [executor.control_socket_path, None]:
        executor.control_socket_path = socket
        argv = executor._build_ssh_command(["git", "status"])
        assert "workstation-alias" in argv
        assert "User=alice" in argv and "Port=2222" in argv
        assert "IdentityFile=/keys/workstation" in argv
        assert "-J" not in argv
        assert not any("ProxyCommand=" in arg for arg in argv)
        assert "StrictHostKeyChecking=no" not in argv


def test_direct_git_transport_preserves_options_without_proxy(tmp_path, monkeypatch):
    manager = _build_remote_manager(tmp_path)
    ctx = manager.context_for("rproj")
    ctx = replace(ctx, settings=replace(ctx.settings, remote=direct()))
    monkeypatch.setattr(manager, "_resolve_remote_path", lambda ctx: "/home/alice/mirrors/rproj")
    manager.executor = RemoteExecutor(human_user="tester", agent_user="tester",
                                     agent_group="tester", logger=logging.getLogger("direct"), login_node="workstation-alias",
                                     control_socket_path="/tmp/test-direct.sock")
    for sock in ["/tmp/test-direct.sock", None]:
        manager.executor.control_socket_path = sock
        url, env = manager._remote_git_env(ctx)
        assert url == "workstation-alias:/home/alice/mirrors/rproj"
        argv = shlex.split(env["GIT_SSH_COMMAND"])
        assert "User=alice" in argv and "Port=2222" in argv
        assert "IdentityFile=/keys/workstation" in argv
        assert "-J" not in argv
        assert not any("ProxyCommand=" in arg for arg in argv)


def test_warm_direct_tunnel_has_no_hostname_probe(connected):
    cfg, settings, calls = connected
    state = session.RemoteSession(mirror_name="tunnel-workstation", login_node="old")
    controls = cli._warm_free_tunnels(settings.remote, state, logging.getLogger("direct"),
                                     debug_ssh=False, config=cfg)
    assert len(calls) == 1
    assert controls == (calls[0],) * 3
    assert state.login_node == settings.remote.host


def test_direct_attach_uses_configured_host_and_options(connected, monkeypatch):
    cfg, settings, calls = connected
    session.RemoteSession(mirror_name="sample", login_node="stale-host").save()
    cfg.mirrors["sample"] = settings
    monkeypatch.setattr(cli, "_get_config", lambda ctx: cfg)
    monkeypatch.setattr(cli, "_get_active_target", lambda ctx: None)
    captured = []
    monkeypatch.setattr(cli.os, "execvp", lambda exe, argv: captured.append(argv))
    ctx = click.Context(click.Command("attach"), obj={})
    cli.attach(ctx, mirror="sample", verbose=False, node=None, via_srun=False)
    argv = captured[0]
    assert "workstation-alias" in argv and "stale-host" not in argv
    assert "User=alice" in argv and "Port=2222" in argv
    assert "-J" not in argv
    assert "tmux attach-session" in argv[-1]


def test_direct_sockets_separate_user_and_port(tmp_path, monkeypatch):
    monkeypatch.setattr(tunnel, "_control_socket_dir", lambda: tmp_path)
    base = direct()
    other_user = replace(base, remote_user="bob")
    other_port = replace(base, ssh_options={"Port": "2200"})
    sockets = [tunnel.SshControl(gateway=r.host, **r.ssh_control_kwargs()).socket_path
               for r in (base, other_user, other_port)]
    assert len(set(sockets)) == 3
    assert tunnel.SshControl(gateway=base.host, **base.ssh_control_kwargs()).socket_path == sockets[0]
    assert tunnel.SshControl(gateway=base.host).socket_path not in sockets


def test_direct_tunnel_commands_do_not_write_cluster_aliases(connected, monkeypatch, capsys):
    from sucoder import sshconfig
    cfg, settings, calls = connected
    monkeypatch.setattr(cli, "_get_config", lambda ctx: cfg)
    monkeypatch.setattr(cli, "_resolve_tunnel_target", lambda ctx: (settings.remote, "workstation"))
    monkeypatch.setattr(tunnel.SshControl, "is_active", lambda *a, **kw: True)
    monkeypatch.setattr(sshconfig, "write_block", lambda *a, **kw: pytest.fail("wrote cluster aliases"))
    monkeypatch.setattr(sshconfig, "block_present", lambda *a, **kw: False)
    ctx = click.Context(click.Command("tunnel"), obj={})
    cli.tunnel_up(ctx, verbose=False, no_config_edit=False)
    assert len(calls) == 1
    capsys.readouterr()
    cli.tunnel_status(ctx, json_output=True)
    result = json.loads(capsys.readouterr().out)
    assert result["hops"] == [{"hop": "host", "host": "workstation-alias", "active": True}]
    cli.tunnel_doctor(ctx)
    assert "no cluster aliases" in capsys.readouterr().out
    closed = []
    monkeypatch.setattr(tunnel.SshControl, "close", lambda self, logger: closed.append(self.socket_path))
    cli.tunnel_down(ctx, prune=False, verbose=False)
    assert closed == [calls[0].socket_path]


def test_direct_forward_and_cancel_use_same_connection(connected, monkeypatch):
    cfg, settings, calls = connected
    monkeypatch.setattr(cli, "_get_config", lambda ctx: cfg)
    monkeypatch.setattr(cli, "_resolve_tunnel_target", lambda ctx: (settings.remote, "workstation"))
    monkeypatch.setattr(cli, "_resolve_forward_node", lambda *a: pytest.fail("looked for compute node"))
    forwards = []
    monkeypatch.setattr(cli, "_mux_forward", lambda *a: (forwards.append(a) or (0, "")))
    ctx = click.Context(click.Command("tunnel"), obj={})
    cli.tunnel_forward(ctx, port=8888, local_port=8889, node=None, cancel=False, verbose=False)
    assert len(calls) == 1
    assert forwards[0] == ("forward", "8889:localhost:8888", str(calls[0].socket_path), "workstation-alias")
    cli.tunnel_forward(ctx, port=8889, local_port=None, node=None, cancel=True, verbose=False)
    assert forwards[1] == ("cancel", *forwards[0][1:])


def test_configured_git_remote_retains_direct_user_and_port(tmp_path):
    manager = _build_remote_manager(tmp_path)
    manager.target_name = "workstation"
    ctx = manager.context_for("rproj")
    ctx = replace(ctx, settings=replace(ctx.settings, remote=direct()))
    manager._configure_target_remote(ctx)
    result = manager.executor.run_human(["git", "remote", "get-url", "workstation"],
                                        cwd=str(ctx.canonical_path))
    assert result.stdout.strip() == f"ssh://alice@workstation-alias:2222/{str(ctx.remote_mirror_path).lstrip('/')}"


def test_direct_liveness_probe_preserves_connection_options(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(tunnel, "_control_socket_dir", lambda: tmp_path)
    remote = direct()
    control = tunnel.SshControl(gateway=remote.host, **remote.ssh_control_kwargs())
    commands = []
    monkeypatch.setattr(tunnel.subprocess, "run", lambda argv, **kw:
                        (commands.append(argv) or SimpleNamespace(returncode=0)))
    assert control._probe_end_to_end()
    argv = commands[0]
    assert "Port=2222" in argv
    assert "IdentityFile=/keys/workstation" in argv
    assert "alice@workstation-alias" in argv


@pytest.mark.parametrize('field,value', [('User', 'bob'), ('HostName', 'other.invalid'), ('Port', '2222')])
def test_direct_socket_tracks_ssh_include_changes(tmp_path, monkeypatch, field, value):
    ssh_config = tmp_path / 'ssh_config'
    original = 'Host direct-regression\n User alice\n HostName example.invalid\n Port 22\n'
    ssh_config.write_text(original)
    import subprocess
    base_config = tmp_path / 'config'
    base_config.write_text(f'Include {ssh_config}\n')
    real_run = subprocess.run
    def isolated_ssh(argv, **kwargs):
        return real_run([argv[0], '-F', str(base_config), *argv[1:]], **kwargs)
    monkeypatch.setattr(subprocess, 'run', isolated_ssh)
    remote = _parse_remote_config({'host': 'direct-regression'})
    before = remote.ssh_control_kwargs()['socket_name']
    old_value = {'User': 'alice', 'HostName': 'example.invalid', 'Port': '22'}[field]
    ssh_config.write_text(original.replace(f'{field} {old_value}', f'{field} {value}'))
    assert remote.ssh_control_kwargs()['socket_name'] != before


@pytest.mark.parametrize('spelling', ['user', 'USER', 'uSeR'])
def test_direct_user_override_matches_openssh(spelling):
    import subprocess
    remote = _parse_remote_config({'host': 'example.invalid', 'remote_user': 'alice',
                                   'ssh_options': {spelling: 'bob'}})
    result = subprocess.run(['ssh', '-G', '-F', '/dev/null',
                             *remote.direct_ssh_options(), remote.host],
                            capture_output=True, text=True, check=True)
    assert 'user alice' in result.stdout.splitlines()
    assert [k for k in remote.direct_ssh_option_map() if k.lower() == 'user'] == ['User']


def test_direct_identity_resolution_failure_does_not_reuse_socket(monkeypatch):
    import subprocess
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(255, 'ssh')
    monkeypatch.setattr(subprocess, 'run', fail)
    with pytest.raises(ConfigError, match='Cannot resolve direct SSH'):
        direct().ssh_control_kwargs()


def test_direct_executor_removes_conflicting_user_option(connected):
    cfg, settings, calls = connected
    settings.remote.ssh_options['uSeR'] = 'bob'
    executor = cli._build_executor(cfg, logging.getLogger('direct'), dry_run=False,
                                   mirror_settings=settings)
    executor.control_socket_path = None
    argv = executor._build_ssh_command(['true'])
    assert 'User=alice' in argv
    assert 'uSeR=bob' not in argv
    assert 'uSeR=bob' not in calls[0].extra_options


def test_direct_collaborate_rejects_node(tmp_path, monkeypatch):
    """--node has no meaning without a scheduler, and _direct_control would
    silently clear the pin; refuse it the way attach and tunnel forward do."""
    from typer.testing import CliRunner

    monkeypatch.setattr(session, "_session_dir", lambda: tmp_path / "sessions")
    path = tmp_path / "config.yaml"
    human = os.environ.get("USER", "coder")
    path.write_text(f"""human_user: {human}
agent_user: {human}
agent_group: {human}
mirror_root: {tmp_path / 'mirrors'}
log_dir: {tmp_path / 'logs'}
targets:
  workstation:
    host: workstation-alias
mirrors:
  sample:
    canonical_repo: {tmp_path / 'canonical'}
""")
    # Startup validation demands the agent can read but not write the config.
    path.chmod(0o444)
    monkeypatch.setattr(cli, "_build_manager_for_mirror",
                        lambda *a, **kw: pytest.fail("bootstrapped a direct target with --node"))

    result = CliRunner().invoke(cli.app, ["--config", str(path), "-T", "workstation",
                                          "collaborate", "sample", "--node", "n0001"])

    assert result.exit_code == 1, (result.output, result.exception)
    assert "--node requires a cluster target." in result.output
    # The pin must not survive as stale session state for the next launch.
    assert not list((tmp_path / "sessions").glob("*")), "wrote session state before refusing"
