"""Cache eligibility rejects stable-but-ineligible native request shapes."""
import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from sucoder import cli, dream, dream_cache_certify as certify, dream_prompt_native as native
from tests.test_dream_prompt import profile


POLICY = "Trusted fixture policy"


def body():
    return dict(model="gpt-6.1-sol", instructions="Native instructions", tools=[],
        input=[dict(type="message", role="developer", content=[dict(type="input_text", text=POLICY)]),
               dict(type="message", role="user", content=[dict(type="input_text", text="untrusted")])])


def test_prefix_binds_native_instructions_cache_mode_and_service_tier():
    original = body()
    initial = certify._prefix(original, "codex", POLICY)
    for key, value in (("instructions", "changed"), ("service_tier", "priority"),
                       ("prompt_cache_key", "new-scope"), ("prompt_cache_options", dict(ttl="12h"))):
        changed = dict(original, **{key: value})
        assert certify._prefix(changed, "codex", POLICY) != initial


@pytest.mark.parametrize("mutation", ["tools", "late_tools", "assistant", "broken_block", "explicit_only", "prewarm", "unknown_model"])
def test_stable_bytes_are_insufficient_without_tool_prohibition_and_eligible_boundary(mutation):
    value = body()
    if mutation == "tools":
        value["tools"] = [dict(type="function", name="external")]
    elif mutation == "late_tools":
        value["input"].append(dict(type="additional_tools", tools=[dict(name="external")]))
    elif mutation == "assistant":
        value["input"][0]["role"] = "assistant"
    elif mutation == "broken_block":
        value["input"].insert(0, dict(type="message", role="assistant", content=[]))
    elif mutation == "unknown_model":
        value["model"] = "unknown-provider-model"
    elif mutation == "explicit_only":
        value["prompt_cache_options"] = dict(mode="explicit")
    else:
        value["prompt_cache_options"] = dict(prewarm=True)
    with pytest.raises(dream.DreamError):
        certify._prefix(value, "codex", POLICY)


def test_claude_needs_the_actual_policy_cache_control_boundary():
    value = dict(model="fixture-model", tools=[dict(name="StructuredOutput")],
        system=[dict(type="text", text=POLICY, cache_control=dict(type="ephemeral"))])
    assert certify._prefix(value, "claude", POLICY)["system"] == value["system"]
    del value["system"][0]["cache_control"]
    with pytest.raises(dream.DreamError, match="cache boundary"):
        certify._prefix(value, "claude", POLICY)


def test_fixture_limits_apply_before_expansion_and_across_requests():
    server = certify.RecordingServer()
    try:
        with pytest.raises(ValueError, match="Compressed requests"):
            server.record(b"compressed", "gzip", "http")
        with pytest.raises(ValueError, match="allowance"):
            server.record(b"x" * 2097153, "", "http")
        server.request_bytes = 4194304
        with pytest.raises(ValueError, match="ceiling"):
            server.record(b"{}", "", "http")
    finally:
        server.server.server_close()


def test_catalog_covers_current_production_policies_and_forced_finish():
    from sucoder import dream_roles
    catalog = native.variants()
    assert len(catalog) == 14
    assert len({(row["policy"], dream.digest(dream.canonical(row["schema"]))) for row in catalog}) == 14
    for row in catalog:
        dream_roles.validate_schema(certify.fixture_result(row["schema"]), row["schema"])
        if row["name"].endswith("-finish"):
            assert row["schema"]["properties"]["action"]["enum"] == ["finish"]


def test_uncertified_policy_or_schema_is_rejected_before_launch():
    value = profile()
    with pytest.raises(dream.DreamError, match="variant"):
        native.manifest(value, "Uncovered trusted policy", b"{}", {})
    del value["certification"]
    with pytest.raises(dream.DreamError, match="certification"):
        native.validate_profile(value)


def test_offline_certificate_and_raw_usage_are_bound_to_profile(tmp_path, monkeypatch):
    frozen = profile("claude")
    catalog = [dict(name="fixture", policy=POLICY, schema=certify.SCHEMA)]
    monkeypatch.setattr(native, "variants", lambda: catalog)
    monkeypatch.setattr(native, "resolve", lambda *args: frozen["build"])
    monkeypatch.setattr(native, "authentication", lambda *args: frozen["authentication"])
    class Boundary:
        network_fd = 0
        def __init__(self, *args):
            pass
        def allow(self, *args):
            pass
        def close(self):
            pass
    monkeypatch.setattr(certify, "Boundary", Boundary)
    original_server = certify.RecordingServer
    servers = []
    def server():
        value = original_server()
        servers.append(value)
        return value
    monkeypatch.setattr(certify, "RecordingServer", server)
    def capture(arguments, data, scratch, env, deadline, boundary, **kwargs):
        if arguments[-1] == "--version":
            return 0, b"fixture-native-1\n", b""
        servers[-1].record(dream.canonical(dict(model="fixture-model", messages=[dict(role="user", content=data.decode())],
            tools=[dict(name="StructuredOutput")], system=[dict(type="text", text=POLICY, cache_control=dict(type="ephemeral"))])), "", "http")
        result = dict(type="result", subtype="success", is_error=False, structured_output=dict(answer="fixture"),
            usage=dict(input_tokens=100, cache_read_input_tokens=30, cache_creation_input_tokens=20, output_tokens=5),
            total_cost_usd=.01)
        return 0, dream.canonical(result) + b"\n", b""
    monkeypatch.setattr(certify.dream_native, "capture", capture)
    config = SimpleNamespace(log_dir=tmp_path / "new-log-root")
    result = certify.certify(config, "claude", "fixture-model")
    assert result["state"] == "certified" and result["paid_inference"] is False
    loaded = native.load_profile(config, "claude", "fixture-model", 999999999999)
    native.manifest(loaded, POLICY, b"{}", certify.SCHEMA)
    report = json.loads((native.directory(config) / ("probe-" + loaded["certification"]["report_sha256"] + ".json")).read_bytes())
    assert report["native_invocations"] == 2 and len(report["observations"]) == 2
    assert report["observations"][0]["raw_usage"]["native"]["cache_creation_input_tokens"] == 20
    assert report["observations"][0]["mapping"]["cost"]["basis"] == "native-list-estimate"
    monkeypatch.setattr(native, "MAX_PROFILE_BYTES", 1)
    second = SimpleNamespace(log_dir=tmp_path / "bounded-profile-root")
    with pytest.raises(dream.DreamError, match="profile exceeds"):
        certify.certify(second, "claude", "fixture-model")
    assert all(path.name.startswith("probe-") for path in native.directory(second).iterdir())


def test_cli_offline_certification_dispatches_without_a_dream_job(monkeypatch):
    calls = []
    def fake(config, harness, model, **kwargs):
        calls.append((harness, model, kwargs))
        return dict(state="uncertified", paid_inference=False, failures=["fixture"])
    monkeypatch.setattr(certify, "certify", fake)
    result = CliRunner().invoke(cli.app, ["dream", "certify-cache", "--harness", "codex", "--model", "fixture"])
    assert result.exit_code == 0, result.output
    assert calls == [("codex", "fixture", dict(seconds=120, route=None))]
