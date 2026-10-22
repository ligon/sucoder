import ast
import json
from pathlib import Path
import sqlite3
import uuid

import pytest

from sucoder import codex_state_runner as runner
from sucoder.codex_remote import build_remote_command
from sucoder.config import AgentLauncher, AgentType, CodexStateConfig
from tests.test_remote_control import _manager


SCHEMA = """CREATE TABLE remote_control_enrollments (
websocket_url TEXT, account_id TEXT, app_server_client_name TEXT,
server_id TEXT, environment_id TEXT, server_name TEXT, updated_at INTEGER,
remote_control_enabled INTEGER);
INSERT INTO remote_control_enrollments VALUES ('wss://test', 'test-account', '', 'old-server', 'old-env', 'old-node', 1, 1);
CREATE TABLE conversations (id TEXT);
INSERT INTO conversations VALUES ('keep-this-thread');
"""


def make_home(path):
    path.mkdir()
    (path / "installation_id").write_text(str(uuid.uuid4()))
    (path / "sessions").mkdir()
    return path


def test_two_owners_get_distinct_ids_and_do_not_modify_shared_enrollment_or_conversations(tmp_path):
    shared = make_home(tmp_path / "shared")
    shared_id = (shared / "installation_id").read_bytes()
    with sqlite3.connect(shared / "state_5.sqlite") as source:
        source.executescript(SCHEMA)
        for name in ("one", "two"):
            local = tmp_path / name
            (local / "db").mkdir(parents=True)
            (local / "home").mkdir()
            with sqlite3.connect(local / "db/state_5.sqlite") as target:
                source.backup(target)
    identities = []
    for name in ("one", "two"):
        local = tmp_path / name
        identity, (private, target) = runner.prepare_remote_identity(str(local / "db"), str(local / "home"), str(shared))
        identities.append(identity)
        assert str(uuid.UUID(identity)) == identity
        assert Path(private).read_text() == identity
        assert target == str(shared / "installation_id")
        with sqlite3.connect(local / "db/state_5.sqlite") as database:
            assert database.execute("SELECT count(*) FROM remote_control_enrollments").fetchone() == (0,)
            assert database.execute("SELECT * FROM conversations").fetchall() == [("keep-this-thread",)]
    assert identities[0] != identities[1]
    assert (shared / "installation_id").read_bytes() == shared_id
    with sqlite3.connect(shared / "state_5.sqlite") as database:
        assert database.execute("SELECT server_id FROM remote_control_enrollments").fetchall() == [("old-server",)]


def test_unknown_cache_schema_refuses_without_removing_rows(tmp_path):
    shared = make_home(tmp_path / "shared")
    (tmp_path / "db").mkdir()
    (tmp_path / "home").mkdir()
    with sqlite3.connect(tmp_path / "db/state_5.sqlite") as database:
        database.executescript("CREATE TABLE remote_control_enrollments (unexpected TEXT); INSERT INTO remote_control_enrollments VALUES ('keep');")
    with pytest.raises(runner.Refused, match="unrecognized"):
        runner.prepare_remote_identity(str(tmp_path / "db"), str(tmp_path / "home"), str(shared))
    with sqlite3.connect(tmp_path / "db/state_5.sqlite") as database:
        assert database.execute("SELECT * FROM remote_control_enrollments").fetchall() == [("keep",)]


def test_remote_without_local_tiering_reuses_state_engine_and_explicit_optout_refuses(tmp_path, monkeypatch):
    manager, ctx = _manager(tmp_path, monkeypatch, remote=True)
    ctx.settings.agent_launcher = AgentLauncher(command=["codex"])
    manager._detected_agent_type = AgentType.CODEX
    state = manager._codex_state_launch(ctx, remote_control=True)
    assert state.temporary_local_root and state.remote_control
    assert state.durable_root.endswith("/.sucoder-state")
    assert manager._codex_state_launch(ctx) is None
    ctx.settings.remote.slurm.codex_state = CodexStateConfig(enabled=False)
    from sucoder.mirror import MirrorError
    with pytest.raises(MirrorError, match="requires isolated"):
        manager._codex_state_launch(ctx, remote_control=True)


def test_legacy_service_uses_owned_app_server_and_does_not_strip_option_values():
    command = build_remote_command(["codex", "-c", 'model="remote-control"', "remote-control", "--no-daemon",
                                    "-c", 'approval_policy="never"'], "server", legacy_service=True)
    at = command.index("sucoder-codex-remote")
    count = int(command[at + 2])
    server = command[at + 3:at + 3 + count]
    assert server[:3] == ["codex", "app-server", "--remote-control"]
    assert '--no-daemon' not in server
    assert 'model="remote-control"' in server


def test_remote_helper_is_standalone_and_compatible_with_target_python36():
    source = Path(__file__).parents[1] / "sucoder/codex_remote_client.py"
    ast.parse(source.read_text(), feature_version=(3, 6))


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_shared_database_links_are_never_sanitized(tmp_path, kind):
    import os
    shared = make_home(tmp_path / "shared")
    with sqlite3.connect(shared / "state_5.sqlite") as database:
        database.executescript(SCHEMA)
    (tmp_path / "db").mkdir()
    (tmp_path / "home").mkdir()
    target = tmp_path / "db/state_5.sqlite"
    if kind == "symlink":
        target.symlink_to(shared / "state_5.sqlite")
    else:
        os.link(shared / "state_5.sqlite", target)
    with pytest.raises(runner.Refused, match="private database"):
        runner.prepare_remote_identity(str(tmp_path / "db"), str(tmp_path / "home"), str(shared))
    with sqlite3.connect(shared / "state_5.sqlite") as database:
        assert database.execute("SELECT server_id FROM remote_control_enrollments").fetchall() == [("old-server",)]
