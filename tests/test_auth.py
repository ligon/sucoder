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

    def __init__(self, pass_values=None):
        self.pass_values = pass_values or {}
        self.calls = []

    def run_human(self, args, **kwargs):
        self.calls.append(list(args))
        entry = args[-1]
        value = self.pass_values.get(entry, "")
        return subprocess.CompletedProcess(
            args, 0 if value else 1, stdout=value + "\n" if value else "", stderr="",
        )

    # run_human returns a CompletedProcess-like; auth reads .returncode/.stdout


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


def test_codex_to_apikey_backs_up_and_logs_in(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    auth_json = codex_home / "auth.json"
    auth_json.write_text('{"tokens": "subscription"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    config = _make_config(tmp_path, openai="openai.com/apikey")
    config.agent_launcher = None
    executor = _FakeExecutor({"openai.com/apikey": "sk-live-key"})
    cfg = HarnessAuthConfig(mode="apikey", apikey_credential="openai")

    login_calls = []

    def fake_run(argv, **kwargs):
        login_calls.append((argv, kwargs))
        return _completed(argv, 0, stdout="Logged in")

    monkeypatch.setattr(subprocess, "run", fake_run)
    logger = logging.getLogger("test")

    auth.codex_to_apikey(config, executor, cfg, logger)

    # Backup written locally (no subscription_credential configured).
    assert (codex_home / "auth.json.sucoder-bak").read_text() == '{"tokens": "subscription"}'
    # Key piped via stdin to codex login --with-api-key, never argv.
    login = login_calls[0]
    assert login[0][:3] == ["codex", "login", "--with-api-key"]
    assert login[1]["input"] == "sk-live-key\n"
    assert "sk-live-key" not in " ".join(login[0])


def test_codex_to_apikey_stores_backup_in_pass_when_configured(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text('{"tokens": "sub"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    config = _make_config(
        tmp_path, openai="openai.com/apikey", **{"codex-chatgpt-auth": "codex/chatgpt-auth.json"}
    )
    executor = _FakeExecutor({"openai.com/apikey": "sk-key"})
    cfg = HarnessAuthConfig(
        mode="apikey", apikey_credential="openai",
        subscription_credential="codex-chatgpt-auth",
    )

    inserts = []

    def fake_insert(executor, entry, value):
        inserts.append((entry, value))

    monkeypatch.setattr(auth, "_run_pass_insert", fake_insert)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _completed(argv, 0))

    auth.codex_to_apikey(config, executor, cfg, logging.getLogger("test"))

    assert inserts == [("codex/chatgpt-auth.json", '{"tokens": "sub"}')]
    assert not (codex_home / "auth.json.sucoder-bak").exists()


def test_codex_to_subscription_restores_from_bak(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json.sucoder-bak").write_text('{"tokens": "sub"}', encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    config = _make_config(tmp_path)
    executor = _FakeExecutor({})
    cfg = HarnessAuthConfig()

    auth.codex_to_subscription(config, executor, cfg, logging.getLogger("test"))

    auth_json = codex_home / "auth.json"
    assert auth_json.read_text() == '{"tokens": "sub"}'
    assert (auth_json.stat().st_mode & 0o777) == 0o600


def test_codex_to_subscription_restores_from_pass(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    config = _make_config(tmp_path, **{"codex-chatgpt-auth": "codex/chatgpt-auth.json"})
    executor = _FakeExecutor({"codex/chatgpt-auth.json": '{"tokens": "sub"}'})
    cfg = HarnessAuthConfig(subscription_credential="codex-chatgpt-auth")

    auth.codex_to_subscription(config, executor, cfg, logging.getLogger("test"))

    assert (codex_home / "auth.json").read_text() == '{"tokens": "sub"}'


def test_codex_to_subscription_without_backup_raises(tmp_path, monkeypatch):
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    config = _make_config(tmp_path)
    executor = _FakeExecutor({})
    with pytest.raises(auth.AuthError, match="No codex subscription backup"):
        auth.codex_to_subscription(config, executor, HarnessAuthConfig(), logging.getLogger("test"))


def test_codex_status_parses_chatgpt(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _completed(argv, 0, stdout="Logged in using ChatGPT\n"),
    )
    st = auth.codex_status(_FakeExecutor())
    assert st.mode == "subscription"


def test_codex_status_parses_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kw: _completed(argv, 0, stdout="Logged in using API key\n"),
    )
    st = auth.codex_status(_FakeExecutor())
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
