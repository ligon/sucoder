"""Tests for the subscription <-> API-key auth switch (sucoder/auth.py)."""

import json
import logging
import os
import subprocess
from pathlib import Path

import pytest

import sucoder.auth as auth
from sucoder.config import Config, CredentialConfig, HarnessAuthConfig


def _make_config(tmp_path, **credentials):
    config = Config(
        human_user="ligon",
        agent_user="coder",
        agent_group="coder",
        mirror_root=tmp_path / "mirrors",
        log_dir=None,
    )
    for name, entry in credentials.items():
        config.credentials[name] = CredentialConfig(pass_entry=entry)
    return config


class _FakeExecutor:
    human_user = "ligon"
    agent_user = "coder"

    def __init__(self, pass_values=None, agent_files=None):
        self.pass_values = pass_values or {}
        # Agent-side virtual filesystem: path -> contents.
        self.agent_files = dict(agent_files or {})
        self.human_calls = []
        self.agent_calls = []
        self.agent_writes = {}

    def run_human(self, args, **kwargs):
        self.human_calls.append(list(args))
        entry = args[-1]
        value = self.pass_values.get(entry, "")
        return subprocess.CompletedProcess(
            args, 0 if value else 1, stdout=value + "\n" if value else "", stderr="",
        )

    def run_agent(self, args, **kwargs):
        args = list(args)
        self.agent_calls.append(args)
        # `cat <path>` reads from the agent's virtual FS.
        if args[:1] == ["cat"]:
            content = self.agent_files.get(args[1])
            if content is None:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout=content, stderr="")
        # `sh -c <writer> <name> <path>` with input= writes to the agent FS.
        if args[:2] == ["sh", "-c"]:
            path = args[-1]
            data = kwargs.get("input", "")
            self.agent_files[path] = data
            self.agent_writes[path] = data
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        # `codex login status` reports based on the stored auth.json.
        if args[:3] == ["codex", "login", "status"]:
            content = self.agent_files.get("/home/coder/.codex/auth.json", "")
            if "OPENAI_API_KEY" in content:
                return subprocess.CompletedProcess(args, 0, stdout="Logged in using API key\n", stderr="")
            if content.strip():
                return subprocess.CompletedProcess(args, 0, stdout="Logged in using ChatGPT\n", stderr="")
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


def _completed(args, rc=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, rc, stdout=stdout, stderr=stderr)


def test_pass_show_reads_first_line(tmp_path):
    executor = _FakeExecutor({"openai.com/apikey": "sk-first\nsk-second"})
    assert auth._pass_show(executor, "openai.com/apikey") == "sk-first"


def test_pass_show_missing_entry_raises(tmp_path):
    executor = _FakeExecutor({})
    with pytest.raises(auth.AuthError, match="Could not read pass entry"):
        auth._pass_show(executor, "nope")


def test_credential_entry_requires_defined_credential(tmp_path):
    config = _make_config(tmp_path)
    with pytest.raises(auth.AuthError, match="not defined under `credentials:`"):
        auth._credential_entry(config, "openai")


_AGENT_HOME = "/home/coder/.codex"


def test_codex_to_apikey_writes_agent_auth_json_via_stdin(tmp_path):
    # Key read as human (pass), auth.json written as agent; key never in argv.
    config = _make_config(tmp_path, openai="openai.com/apikey")
    executor = _FakeExecutor(
        {"openai.com/apikey": "sk-live-key"},
        agent_files={f"{_AGENT_HOME}/auth.json": '{"tokens": "subscription"}'},
    )
    cfg = HarnessAuthConfig(mode="apikey", apikey_credential="openai")

    auth.codex_to_apikey(config, executor, cfg, logging.getLogger("test"))

    written = executor.agent_writes[f"{_AGENT_HOME}/auth.json"]
    payload = json.loads(written)
    assert payload["OPENAI_API_KEY"] == "sk-live-key"
    assert payload["auth_mode"] == "apikey"
    # The subscription login was backed up locally (no subscription_credential).
    assert executor.agent_files[f"{_AGENT_HOME}/auth.json.sucoder-bak"] == '{"tokens": "subscription"}'
    # The key never appears on any command line, only via stdin to the writer.
    for call in executor.agent_calls:
        assert "sk-live-key" not in " ".join(call)


def test_codex_to_apikey_stores_backup_in_pass_when_configured(tmp_path, monkeypatch):
    config = _make_config(
        tmp_path, openai="openai.com/apikey", **{"codex-chatgpt-auth": "codex/chatgpt-auth.json"}
    )
    executor = _FakeExecutor(
        {"openai.com/apikey": "sk-key"},
        agent_files={f"{_AGENT_HOME}/auth.json": '{"tokens": "sub"}'},
    )
    cfg = HarnessAuthConfig(
        mode="apikey", apikey_credential="openai",
        subscription_credential="codex-chatgpt-auth",
    )

    inserts = []
    monkeypatch.setattr(auth, "_run_pass_insert", lambda ex, entry, value: inserts.append((entry, value)))

    auth.codex_to_apikey(config, executor, cfg, logging.getLogger("test"))

    assert inserts == [("codex/chatgpt-auth.json", '{"tokens": "sub"}')]
    assert f"{_AGENT_HOME}/auth.json.sucoder-bak" not in executor.agent_files


def test_codex_to_apikey_no_existing_login_skips_backup(tmp_path):
    config = _make_config(tmp_path, openai="openai.com/apikey")
    executor = _FakeExecutor({"openai.com/apikey": "sk-key"})
    cfg = HarnessAuthConfig(mode="apikey", apikey_credential="openai")

    auth.codex_to_apikey(config, executor, cfg, logging.getLogger("test"))

    assert f"{_AGENT_HOME}/auth.json.sucoder-bak" not in executor.agent_files
    assert json.loads(executor.agent_writes[f"{_AGENT_HOME}/auth.json"])["OPENAI_API_KEY"] == "sk-key"


def test_codex_to_subscription_restores_from_bak(tmp_path):
    config = _make_config(tmp_path)
    executor = _FakeExecutor(
        agent_files={f"{_AGENT_HOME}/auth.json.sucoder-bak": '{"tokens": "sub"}'},
    )

    auth.codex_to_subscription(config, executor, HarnessAuthConfig(), logging.getLogger("test"))

    assert executor.agent_writes[f"{_AGENT_HOME}/auth.json"] == '{"tokens": "sub"}'


def test_codex_to_subscription_restores_from_pass(tmp_path):
    config = _make_config(tmp_path, **{"codex-chatgpt-auth": "codex/chatgpt-auth.json"})
    executor = _FakeExecutor({"codex/chatgpt-auth.json": '{"tokens": "sub"}'})
    cfg = HarnessAuthConfig(subscription_credential="codex-chatgpt-auth")

    auth.codex_to_subscription(config, executor, cfg, logging.getLogger("test"))

    assert executor.agent_writes[f"{_AGENT_HOME}/auth.json"] == '{"tokens": "sub"}'


_APIKEY_BLOB = json.dumps({"auth_mode": "apikey", "OPENAI_API_KEY": "sk-old"})


@pytest.mark.parametrize("blob", [
    _APIKEY_BLOB, '{"OPENAI_API_KEY": "sk-old"}', '{"auth_mode": "chatgpt", "tokens": null}',
    '{"auth_mode": "apikey", "tokens": {"access_token": "stale"}}', "[]", "not json",
])
def test_codex_second_apikey_switch_keeps_subscription_backup(tmp_path, blob):
    config = _make_config(tmp_path, openai="openai.com/apikey")
    executor = _FakeExecutor(
        {"openai.com/apikey": "sk-new"},
        agent_files={
            f"{_AGENT_HOME}/auth.json": blob,
            f"{_AGENT_HOME}/auth.json.sucoder-bak": '{"tokens": "sub"}',
        },
    )
    auth.codex_to_apikey(config, executor, HarnessAuthConfig(mode="apikey", apikey_credential="openai"),
                         logging.getLogger("test"))
    assert executor.agent_files[f"{_AGENT_HOME}/auth.json.sucoder-bak"] == '{"tokens": "sub"}'
    assert json.loads(executor.agent_files[f"{_AGENT_HOME}/auth.json"])["OPENAI_API_KEY"] == "sk-new"


def test_codex_second_apikey_switch_keeps_pass_backup(tmp_path, monkeypatch):
    config = _make_config(
        tmp_path, openai="openai.com/apikey", **{"codex-chatgpt-auth": "codex/chatgpt-auth.json"}
    )
    executor = _FakeExecutor(
        {"openai.com/apikey": "sk-new"}, agent_files={f"{_AGENT_HOME}/auth.json": _APIKEY_BLOB},
    )
    monkeypatch.setattr(auth, "_run_pass_insert", lambda *a: pytest.fail("overwrote pass backup"))
    cfg = HarnessAuthConfig(
        mode="apikey", apikey_credential="openai", subscription_credential="codex-chatgpt-auth",
    )
    auth.codex_to_apikey(config, executor, cfg, logging.getLogger("test"))


def test_codex_to_subscription_refuses_apikey_backup(tmp_path):
    current = '{"tokens": "current"}'
    executor = _FakeExecutor(agent_files={
        f"{_AGENT_HOME}/auth.json": current, f"{_AGENT_HOME}/auth.json.sucoder-bak": _APIKEY_BLOB,
    })
    with pytest.raises(auth.AuthError, match="holds no subscription login"):
        auth.codex_to_subscription(_make_config(tmp_path), executor, HarnessAuthConfig(), logging.getLogger("test"))
    assert executor.agent_files[f"{_AGENT_HOME}/auth.json"] == current


def test_codex_to_subscription_without_backup_raises(tmp_path):
    config = _make_config(tmp_path)
    executor = _FakeExecutor()
    with pytest.raises(auth.AuthError, match="No codex subscription backup"):
        auth.codex_to_subscription(config, executor, HarnessAuthConfig(), logging.getLogger("test"))


_AUTH = f"{_AGENT_HOME}/auth.json"
_BAK = f"{_AGENT_HOME}/auth.json.sucoder-bak"


@pytest.mark.parametrize("blob", [_APIKEY_BLOB, '{"OPENAI_API_KEY": "sk-old"}'])
def test_codex_ensure_subscription_restores_over_apikey(tmp_path, blob):
    executor = _FakeExecutor(agent_files={_AUTH: blob, _BAK: '{"tokens": "sub"}'})
    assert auth.codex_ensure_subscription(
        _make_config(tmp_path), executor, HarnessAuthConfig(), logging.getLogger("test"))
    assert executor.agent_files[_AUTH] == '{"tokens": "sub"}'


@pytest.mark.parametrize("blob", [
    None, '{"tokens": "current"}', '{"auth_mode": "chatgpt", "tokens": {"a": "b"}, "OPENAI_API_KEY": "sk-x"}',
    '{"auth_mode": "chatgpt", "tokens": null}', "[]", "not json",
])
def test_codex_ensure_subscription_leaves_anything_else(tmp_path, blob):
    files = {_BAK: '{"tokens": "older"}'}
    if blob is not None:
        files[_AUTH] = blob
    executor = _FakeExecutor(agent_files=files)
    assert not auth.codex_ensure_subscription(
        _make_config(tmp_path), executor, HarnessAuthConfig(), logging.getLogger("test"))
    assert executor.agent_writes == {}


@pytest.mark.parametrize("backup", [None, _APIKEY_BLOB])
def test_codex_ensure_subscription_without_login_raises(tmp_path, backup):
    files = {_AUTH: _APIKEY_BLOB}
    if backup is not None:
        files[_BAK] = backup
    executor = _FakeExecutor(agent_files=files)
    with pytest.raises(auth.AuthError):
        auth.codex_ensure_subscription(
            _make_config(tmp_path), executor, HarnessAuthConfig(), logging.getLogger("test"))
    assert executor.agent_writes == {}


def test_codex_apikey_launch_then_subscription_round_trip(tmp_path):
    config = _make_config(tmp_path, openai="openai.com/apikey")
    executor = _FakeExecutor({"openai.com/apikey": "sk-new"}, agent_files={_AUTH: '{"tokens": "sub"}'})
    log = logging.getLogger("test")
    for _ in range(2):  # a repeated API-key launch must not cost the login
        auth.codex_to_apikey(config, executor, HarnessAuthConfig(), log)
    assert auth.codex_ensure_subscription(config, executor, HarnessAuthConfig(), log)
    assert executor.agent_files[_AUTH] == '{"tokens": "sub"}'


def test_codex_status_parses_chatgpt(tmp_path):
    config = _make_config(tmp_path)
    executor = _FakeExecutor(agent_files={f"{_AGENT_HOME}/auth.json": '{"tokens": "sub"}'})
    st = auth.codex_status(executor, config)
    assert st.mode == "subscription"


def test_codex_status_parses_api_key(tmp_path):
    config = _make_config(tmp_path)
    executor = _FakeExecutor(
        agent_files={f"{_AGENT_HOME}/auth.json": '{"auth_mode": "apikey", "OPENAI_API_KEY": "sk"}'},
    )
    st = auth.codex_status(executor, config)
    assert st.mode == "apikey"


def test_claude_status_parses_json(tmp_path, monkeypatch):
    payload = {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"}
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _completed(argv, 0, stdout=json.dumps(payload)),
    )
    st = auth.claude_status(_FakeExecutor())
    assert st.mode == "subscription"


def test_claude_status_api_key(tmp_path, monkeypatch):
    payload = {"loggedIn": True, "authMethod": "api_key", "apiKeySource": "ANTHROPIC_API_KEY"}
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _completed(argv, 0, stdout=json.dumps(payload)),
    )
    st = auth.claude_status(_FakeExecutor())
    assert st.mode == "apikey"


def test_claude_to_apikey_requires_credential(tmp_path):
    config = _make_config(tmp_path)
    with pytest.raises(auth.AuthError, match="not defined under `credentials:`"):
        auth.to_apikey(config, _FakeExecutor(), "claude", logging.getLogger("test"))


def test_claude_to_subscription_is_a_noop(tmp_path, caplog):
    config = _make_config(tmp_path)
    with caplog.at_level(logging.INFO):
        auth.to_subscription(config, _FakeExecutor(), "claude", logging.getLogger("test"))
    assert "untouched" in caplog.text


def test_unknown_harness_raises(tmp_path):
    config = _make_config(tmp_path)
    with pytest.raises(auth.AuthError, match="Unknown harness"):
        auth.status(config, _FakeExecutor(), "gemini")
