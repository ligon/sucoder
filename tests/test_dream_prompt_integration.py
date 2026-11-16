"""Stable delivery through the real coordinator, journal and offline replay."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import pytest

from sucoder import dream, dream_explore, dream_native, dream_prompt, dream_prompt_native
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, options
from tests.test_dream_prompt import profile
from tests.test_dream_prompt import task as prompt_task
from tests.test_dream_wiki import planned


def certified(harness):
    value = profile(harness)
    value.update(version=ExploringTransport.version, model="explicit-model")
    value["variants"] = [dict(name=row["name"], policy_sha256=dream.digest(row["policy"].encode()),
        schema_sha256=dream.digest(dream.canonical(row["schema"])), prefix_sha256="d" * 64)
        for row in dream_prompt_native.variants()]
    value["certification"]["native_invocations"] = 2 * len(value["variants"])
    value.pop("sha256")
    value["sha256"] = dream.digest(dream.canonical(value))
    return value


class StableTransport(ExploringTransport):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.requests, self.preflights = [], []

    def preflight(self, request, deadline):
        self.preflights.append(request["sha256"])

    def __call__(self, harness, model, task, schema, deadline, *, request):
        dream_prompt.verify(request, task, request["profile"], schema)
        self.requests.append(copy.deepcopy(request))
        result, usage, trace = super().__call__(harness, model, task, schema, deadline)
        trace.update(contract="native-prompt-trace-v1", assignment_sha256=dream.digest(dream.canonical(task)),
            request_sha256=request["sha256"],
            delivery_manifest_sha256=dream.digest(dream.canonical(request["manifest"])),
            usage_mapping=request["profile"]["usage_mapping"], delivery={key: request["manifest"][key]
                for key in ("policy_sha256", "stdin_sha256", "schema_sha256", "profile_sha256")})
        return result, usage, trace


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_stable_roles_share_wallet_and_replay_without_build_auth_or_provider(workspace, monkeypatch, harness):
    repo, _, config, _ = workspace
    planned(workspace)
    frozen = certified(harness)
    monkeypatch.setattr(dream_prompt_native, "load_profile", lambda *args: frozen)
    transport = StableTransport()
    result = dream_explore.start(config, cwd=repo, overrides=options(harness, prompt_layout="stable"), transport=transport)
    assert result["state"] == "published"
    assert [row["assignment"]["role"] for row in transport.requests] == ["explorer"] * 4 + ["author", "reviewer"]
    assert len(transport.preflights) == len(transport.requests) == 6
    job = Path(result["job_path"])
    budget = dream_explore.budget_status(config, job)
    assert budget["reserved_calls"] == 6
    assert budget["normalized_usage"]["calls_missing"]["input_total"] == 6
    assert budget["normalized_usage"]["by_role"]["reviewer"]["calls"] == 1
    for request in transport.requests:
        matches = list(job.rglob("*" + request["assignment"]["assignment_id"] + "-request.json"))
        # Editorial envelopes are in the selected plan, discovery in the job.
        if not matches:
            matches = list(Path(result["plan_path"]).parent.glob("*" + request["assignment"]["assignment_id"] + "-request.json"))
        assert len(matches) == 1 and json.loads(matches[0].read_bytes()) == request
    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only replay touched native build, credentials or provider")
    for name in ("load_profile", "authentication", "resolve", "Transport"):
        monkeypatch.setattr(dream_prompt_native, name, forbidden)
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    changed = replace(config, dream=replace(config.dream, prompt_layout="legacy"))
    assert dream_explore.start(changed, job) == result
    assert dream_explore.status(changed, job)["state"] == "published"


@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_replay_refuses_changed_request_without_native_startup(workspace, monkeypatch, fault):
    repo, _, config, _ = workspace
    planned(workspace)
    monkeypatch.setattr(dream_prompt_native, "load_profile", lambda *args: certified("codex"))
    transport = StableTransport()
    result = dream_explore.start(config, cwd=repo, overrides=options(prompt_layout="stable"), transport=transport)
    job = Path(result["job_path"])
    request = next(job.glob("*-request.json"))
    if fault == "missing":
        request.unlink()
    else:
        value = json.loads(request.read_bytes())
        value["plan"]["current"]["value"]["budget"]["remaining_calls"] = 1000
        value.pop("sha256")
        value["sha256"] = dream.digest(dream.canonical(value))
        request.write_bytes(dream.canonical(value))
    def forbidden(*args, **kwargs):
        raise AssertionError("Corrupt request triggered provider startup")
    monkeypatch.setattr(dream_prompt_native, "Transport", forbidden)
    with pytest.raises(dream.DreamError):
        dream_explore.status(config, job)
    assert len(transport.calls) == 6


def test_failed_stable_invocation_cannot_repeat_its_reservation(workspace, monkeypatch):
    repo, _, config, _ = workspace
    monkeypatch.setattr(dream_prompt_native, "load_profile", lambda *args: certified("codex"))
    transport = StableTransport(fail=True)
    job = dream_explore.prepare(config, cwd=repo, overrides=options(prompt_layout="stable"))["job_path"]
    with pytest.raises(dream.DreamError):
        dream_explore.run(config, job, transport=transport)
    with pytest.raises(dream.DreamError):
        dream_explore.run(config, job, transport=transport)
    assert len(transport.calls) == 1
    assert dream_explore.budget_status(config, job)["reserved_calls"] == 1


def test_staged_account_switch_is_rejected_even_if_original_switches_back(tmp_path, monkeypatch):
    original = tmp_path / "original"
    original.mkdir()
    (original / "auth.json").write_text(json.dumps(dict(tokens=dict(account_id="A"))))
    monkeypatch.setenv("CODEX_HOME", str(original))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    frozen = profile()
    frozen["authentication"] = dream_prompt_native.authentication("codex")
    legacy_environment = dream_native.environment
    def race(scratch, harness):
        (original / "auth.json").write_text(json.dumps(dict(tokens=dict(account_id="B"))))
        env = legacy_environment(scratch, harness)
        (original / "auth.json").write_text(json.dumps(dict(tokens=dict(account_id="A"))))
        return env
    monkeypatch.setattr(dream_native, "environment", race)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(dream.DreamError, match="Staged native authentication"):
        dream_prompt_native.environment(frozen, scratch)


def test_explicit_api_key_staging_does_not_copy_competing_subscription_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    frozen = profile()
    frozen["authentication"] = dream_prompt_native.authentication("codex")
    def forbidden(*args):
        raise AssertionError("API-key route copied native subscription credentials")
    monkeypatch.setattr(dream_native, "environment", forbidden)
    env = dream_prompt_native.environment(frozen, tmp_path)
    assert json.loads((Path(env["CODEX_HOME"]) / "auth.json").read_bytes()) == dict(OPENAI_API_KEY="synthetic-test-key")


@pytest.mark.parametrize("duplicate", [False, True])
def test_native_delivery_uses_exact_inline_policy_sealed_schema_and_one_final_aggregate(tmp_path, monkeypatch, duplicate):
    home = tmp_path / "auth"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps(dict(tokens=dict(account_id="fixture-A"))))
    monkeypatch.setenv("CODEX_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    frozen = profile()
    fingerprint = dream_prompt_native.fingerprint(sys.executable, time.monotonic() + 5)
    frozen["build"].update(entry=fingerprint, native=fingerprint)
    frozen["authentication"] = dream_prompt_native.authentication("codex")
    frozen.pop("sha256")
    frozen["sha256"] = dream.digest(dream.canonical(frozen))
    assignment = prompt_task()
    request = dream_prompt.envelope(assignment, frozen, assignment["result_schema"])
    def capture(arguments, data, scratch, env, deadline, boundary, **kwargs):
        assert data == dream_prompt.render(request["plan"])[1]
        argument = next(value for value in arguments if value.startswith("developer_instructions="))
        assert json.loads(argument.split("=", 1)[1]) == dream_prompt.render(request["plan"])[0]
        path = arguments[arguments.index("--output-schema") + 1]
        fd = int(path.rsplit("/", 1)[1])
        assert fd in kwargs["extra_fds"]
        assert Path(path).read_bytes() == dream.canonical(assignment["result_schema"])
        with pytest.raises(PermissionError):
            os.pwrite(fd, b"x", 0)
        final = dict(type="turn.completed", usage=dict(input_tokens=100, cached_input_tokens=30,
            cache_write_input_tokens=20, output_tokens=5))
        events = [dict(type="item.completed", item=dict(type="agent_message", text="{}")), final]
        if duplicate:
            events.append(final)
        return 0, b"\n".join(dream.canonical(value) for value in events), b""
    monkeypatch.setattr(dream_native, "capture", capture)
    result, usage, trace = dream_prompt_native.invoke(frozen, assignment, assignment["result_schema"], request, time.monotonic() + 5)
    dream_prompt.verify_trace(request, trace)
    if duplicate:
        assert result is usage is None and "duplicate" in trace["validation_error"]
    else:
        assert result == {} and usage["native"]["cache_write_input_tokens"] == 20
