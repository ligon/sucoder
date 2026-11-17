"""Body-only native submission, strict bindings and no-repeat recovery."""
import copy
import json
import os
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_editor, dream_explore, dream_native, dream_prompt_native, dream_roles, dream_submission as submit
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, options
from tests.test_dream_wiki import planned


def task():
    inputs = dict(evidence="A qualification is recorded, not a human approval.")
    value = dict(contract="fixture", run_id="a" * 64, role="reviewer", brief="Independently review the evidence.",
        input=inputs, input_sha256=dream.digest(dream.canonical(inputs)), result_schema=dream_editor.schema("reviewer"))
    value["assignment_id"] = dream.digest(dream.canonical(value))
    return submit.assignment(value, submit.CONTRACT)


def body():
    return dict(failures=[], verdict="accept", findings=[], requests=[], task_checks=[])


def trace(value, harness, result=None):
    result = body() if result is None else result
    events = ([dict(type="item.completed", item=dict(type="agent_message", text=json.dumps(result))),
        dict(type="turn.completed", usage=dict(input_tokens=10, cached_input_tokens=0, output_tokens=4))]
        if harness == "codex" else [dict(type="system", subtype="init", tools=["StructuredOutput"], mcp_servers=[]),
            dict(type="result", subtype="success", is_error=False, structured_output=result,
                usage=dict(input_tokens=10, output_tokens=4), total_cost_usd=None)])
    return dict(harness=harness, model="fixture", version="fixture-1", returncode=0,
        input_sha256=dream.digest(dream.canonical(value)), stderr="",
        stdout="\n".join(json.dumps(event) for event in events) + "\n", submission=submit.manifest(value))


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_submission_supplies_frozen_metadata_without_model_copying(harness):
    value = task()
    original = copy.deepcopy(value)
    raw = trace(value, harness)
    result, usage = submit.parse(value, raw)
    assert result == dict(body(), **{key: value[key] for key in submit.FIELDS})
    assert dream_native.parse_trace(raw["stdout"].encode(), harness)[0] == body()
    assert usage["native"]["input_tokens"] == 10
    assert value == original
    assert not set(submit.FIELDS) & submit.wire_schema(value)["properties"].keys()
    assert json.loads(submit.payload(value))["result_schema"] == submit.wire_schema(value)


@pytest.mark.parametrize("harness", ["codex", "claude"])
@pytest.mark.parametrize("fault", ["injected_metadata", "wrong_assignment", "wrong_schema", "wrong_stdin", "missing_manifest", "duplicate_final", "missing_final"])
def test_submission_rejects_unbound_or_malformed_native_reply(harness, fault):
    value = task()
    raw = trace(value, harness, dict(body(), input_sha256="wrong") if fault == "injected_metadata" else None)
    if fault == "wrong_assignment": raw["input_sha256"] = "b" * 64
    if fault == "wrong_schema": raw["submission"]["schema_sha256"] = "b" * 64
    if fault == "wrong_stdin": raw["submission"]["stdin_sha256"] = "b" * 64
    if fault == "missing_manifest": del raw["submission"]
    if fault in ("duplicate_final", "missing_final"):
        events = raw["stdout"].splitlines()
        raw["stdout"] = "\n".join(events + events[-1:] if fault == "duplicate_final" else events[:-1]) + "\n"
    with pytest.raises(dream.DreamError):
        submit.parse(value, raw)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_native_process_receives_body_schema_and_raw_body_is_preserved(tmp_path, monkeypatch, harness):
    executable = tmp_path / harness
    executable.write_text('''#!/usr/bin/python3 -S
import json,sys
if '--version' in sys.argv:
    print('fixture-1'); raise SystemExit
if '--help' in sys.argv:
    print('--ignore-user-config --output-schema --ephemeral --ignore-rules --restricted --safe-mode --json-schema --tools --permission-prompts'); raise SystemExit
value=json.load(sys.stdin)
schema=(json.loads(sys.argv[sys.argv.index('--json-schema')+1]) if '--json-schema' in sys.argv else json.load(open(sys.argv[sys.argv.index('--output-schema')+1])))
assert not set(('assignment_id','input_sha256','role')) & set(schema['properties'])
assert schema==value['result_schema']
result=dict(failures=[],verdict='accept',findings=[],requests=[],task_checks=[])
if '--json-schema' in sys.argv:
    print(json.dumps(dict(type='system',subtype='init',tools=['StructuredOutput'],mcp_servers=[])))
    print(json.dumps(dict(type='result',subtype='success',is_error=False,structured_output=result,usage={})))
else:
    print(json.dumps(dict(type='item.completed',item=dict(type='agent_message',text=json.dumps(result)))))
    print(json.dumps(dict(type='turn.completed',usage={})))
''')
    executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    value = task()
    result, usage, raw = dream_native.invoke(harness, "fixture", value, value["result_schema"], time.monotonic() + 10)
    assert result is not None, raw
    assert result["input_sha256"] == value["input_sha256"]
    assert dream_native.parse_trace(raw["stdout"].encode(), harness)[0] == body()
    assert submit.parse(value, raw) == (result, usage)


@pytest.mark.parametrize("where", ["discovery", "editorial"])
@pytest.mark.parametrize("fault", ["missing", "corrupt"])
def test_completed_submission_replay_refuses_changed_manifest_without_inference(workspace, monkeypatch, where, fault):
    repo, _, config, _ = workspace
    planned(workspace)
    transport = ExploringTransport()
    result = dream_explore.start(config, cwd=repo, overrides=options(), transport=transport)
    assert result["state"] == "published"
    job = Path(result["job_path"])
    directory = job if where == "discovery" else Path(result["plan_path"]).parent
    path = next(directory.glob("*-submission.json"))
    path.unlink() if fault == "missing" else path.write_text("{}")
    count = len(transport.calls)
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay attempted native inference")
    monkeypatch.setattr(dream_native, "Transport", forbidden)
    with pytest.raises(dream.DreamError, match="submission"):
        dream_explore.status(config, job)
    assert len(transport.calls) == count


def test_legacy_reply_with_wrong_checksum_is_not_reinterpreted():
    value = task()
    del value["submission_contract"]
    result = dict(body(), **{key: value[key] for key in submit.FIELDS})
    result["input_sha256"] = "b" * 64
    raw = trace(value, "codex", result)
    del raw["submission"]
    restored, _ = submit.parse(value, raw)
    coordinator = dream_roles.Coordinator(dict(plan_id="fixture"), -1, "codex", "fixture", time.monotonic() + 10)
    with pytest.raises(dream.DreamError, match="input_sha256"):
        coordinator.check_result(value, restored)


def test_failed_submission_is_not_automatically_repeated(workspace):
    repo, _, config, _ = workspace
    transport = ExploringTransport(no_selection=True, fail=True)
    prepared = dream_explore.prepare(config, cwd=repo, overrides=options())
    job = Path(prepared["job_path"])
    for error in ("Native role failed", "earlier role attempt has no validated result"):
        with pytest.raises(dream.DreamError, match=error):
            dream_explore.run(config, job, transport=transport)
    assert len(transport.calls) == 1
    assert len(list(job.glob("*-submission.json"))) == 1
    assert dream_explore.budget_status(config, job)["reserved_calls"] == 1


@pytest.mark.parametrize("contract", [None, "unknown-submission"])
def test_invalid_explicit_submission_contract_is_rejected(contract):
    value = dict(task(), submission_contract=contract)
    with pytest.raises(dream.DreamError, match="contract"):
        submit.wire_schema(value)


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_trace_keeps_the_dispatched_assignment_when_parent_task_changes(monkeypatch, harness):
    value = task()
    dispatched_sha = dream.digest(dream.canonical(value))
    dispatched_manifest = submit.manifest(value)
    monkeypatch.setattr(dream_native.shutil, "which", lambda name: "/usr/bin/true")
    def capture(argv, payload, *args, **kwargs):
        if "--help" in argv:
            return 0, b"--ignore-user-config --output-schema --ephemeral --ignore-rules --restricted --safe-mode --json-schema --tools --permission-prompts", b""
        if "--version" in argv:
            return 0, b"fixture-1", b""
        assert dream.digest(payload) == dispatched_manifest["stdin_sha256"]
        raw = trace(value, harness)
        value["input"]["evidence"] = "Changed after dispatch"
        return 0, raw["stdout"].encode(), b""
    monkeypatch.setattr(dream_native, "capture", capture)
    result, usage, raw = dream_native.invoke(harness, "fixture", value, value["result_schema"], time.monotonic() + 10)
    assert result is usage is None
    assert raw["input_sha256"] == dispatched_sha
    assert raw["submission"] == dispatched_manifest
    assert "validation_error" in raw


@pytest.mark.parametrize("fault", ["legacy_contract", "wrong_catalog"])
def test_new_profile_load_refuses_legacy_submission_or_stale_catalog(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    from tests.test_dream_prompt import profile
    value = profile("claude")
    if fault == "wrong_catalog":
        value["submission_contract"] = submit.CONTRACT
        value.pop("sha256")
        value["sha256"] = dream.digest(dream.canonical(value))
    native = dream_prompt_native
    monkeypatch.setattr(native, "resolve", lambda *a: value["build"])
    monkeypatch.setattr(native, "authentication", lambda *a: value["authentication"])
    config = SimpleNamespace(log_dir=tmp_path / "logs")
    directory = native.directory(config)
    directory.mkdir(mode=0o700)
    key = native.profile_key(value["build"], value["model"], value["authentication"]["route"])
    path = directory / (key + ".json")
    path.write_bytes(dream.canonical(value))
    path.chmod(0o600)
    with pytest.raises(dream.DreamError, match="submission"):
        native.load_profile(config, "claude", value["model"], time.monotonic() + 10)
