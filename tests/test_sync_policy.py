"""Authority, preservation, filtering and concurrency acceptance for issue 66."""
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from sucoder import cli
from sucoder.config import ConfigError, load_config
from sucoder.mirror import MirrorError
from tests.test_cli import _write_config
from tests.test_sync_safety import advance, git, snapshot, topology


def filtered_topology(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = topology(tmp_path, monkeypatch)
    git(satellite, "config", "uploadpack.allowFilter", "true")
    git(satellite, "config", "receive.denyCurrentBranch", "updateInstead")
    return manager, ctx, laptop, satellite


def full_topology(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    git(laptop, "sparse-checkout", "disable")
    git(laptop, "config", "remote.gate.promisor", "false")
    return manager, ctx, laptop, satellite


def offline_object(path, oid):
    return git(path, "-c", "protocol.allow=never", "-c", "protocol.file.allow=never",
               "cat-file", "-e", oid, check=False).returncode == 0


def state(path):
    return (git(path, "rev-parse", "HEAD").stdout,
            git(path, "rev-parse", "refs/heads/main").stdout,
            (path / ".git/index").read_bytes(),
            {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*")
             if p.is_file() and ".git" not in p.relative_to(path).parts})


def test_external_policy_defaults_to_existing_mirror_launch(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    ctx.settings.sync_policy = "external"
    before = snapshot(laptop), snapshot(satellite)
    calls = []
    monkeypatch.setattr(manager, "launch_agent", lambda c, **kw: calls.append(kw) or 0)
    for method in ("prepare_canonical", "ensure_remote_clone", "ensure_clone", "_configure_target_remote"):
        monkeypatch.setattr(manager, method, lambda *a, **k: pytest.fail("mutated repository"))
    assert manager.bootstrap(ctx) == 0
    assert calls[0]["sync"] is False
    assert (snapshot(laptop), snapshot(satellite)) == before


@pytest.mark.parametrize("method, kwargs", [
    ("sync", {}), ("sync", {"allow_unverified_mirror": True}),
    ("ensure_remote_clone", {}), ("ensure_clone", {}), ("prepare_canonical", {}),
    ("_pull_from_remote", {}), ("_pull_from_local", {}), ("_sync_remote", {}),
    ("start_task", {"task_name": "topic"}),
    ("bootstrap", {"sync": True}), ("launch_agent", {"sync": True}),
    ("bootstrap", {"task_name": "topic"}),
])
def test_external_policy_refuses_managed_operations(tmp_path, monkeypatch, method, kwargs):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    ctx.settings.sync_policy = "external"
    before = snapshot(laptop), snapshot(satellite)
    with pytest.raises(MirrorError, match="external"):
        getattr(manager, method)(ctx, **kwargs)
    assert (snapshot(laptop), snapshot(satellite)) == before


@pytest.mark.parametrize("bad", [True, False, None, "", "remote", [], {}])
def test_config_rejects_unknown_authority_policy(tmp_path, bad):
    path = _write_config(tmp_path, skills_entry=tmp_path / "skills")
    raw = yaml.safe_load(path.read_text())
    raw["mirrors"]["sample"]["sync_policy"] = bad
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="sync_policy"):
        load_config(path)


@pytest.mark.parametrize("command", ["collaborate", "agents-run"])
@pytest.mark.parametrize("flag", ["--sync", "--task"])
def test_cli_external_policy_refuses_before_tmux(tmp_path, monkeypatch, command, flag):
    path = _write_config(tmp_path, skills_entry=tmp_path / "skills")
    raw = yaml.safe_load(path.read_text())
    raw["mirrors"]["sample"]["sync_policy"] = "external"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tmux_launch_preamble", lambda *a, **k: pytest.fail("created session"))
    args = ["--config", str(path), command, "sample", flag]
    if flag == "--task":
        args.append("topic")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0
    assert "external" in result.output


@pytest.mark.parametrize("dirty", ["tracked", "staged", "topic", "linked"])
def test_reconciliation_preserves_dirty_or_other_worktrees(tmp_path, monkeypatch, dirty):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    advance(satellite, "remote.txt")
    if dirty == "topic":
        git(laptop, "checkout", "-qb", "work/topic")
    elif dirty == "linked":
        git(laptop, "worktree", "add", "--force", str(tmp_path / "linked"), "main")
    else:
        (laptop / "README.md").write_text("preserve edit\n")
        if dirty == "staged":
            git(laptop, "add", "README.md")
    before = state(laptop)
    with pytest.raises(MirrorError, match="Refusing"):
        manager._pull_from_remote(ctx)
    assert state(laptop) == before
    assert not getattr(manager, "_sync_leases", {})


@pytest.mark.parametrize("ignored", [False, True])
def test_fast_forward_refuses_untracked_and_ignored_collisions(tmp_path, monkeypatch, ignored):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    advance(satellite, "collision.txt")
    if ignored:
        (laptop / ".git/info/exclude").write_text("collision.txt\n")
    (laptop / "collision.txt").write_text("irreplaceable local file\n")
    before = state(laptop)
    with pytest.raises(MirrorError, match="would be overwritten"):
        manager._pull_from_remote(ctx)
    assert state(laptop) == before


@pytest.mark.parametrize("base", ["main", "master"])
def test_explicit_filtered_pull_does_not_download_excluded_blobs(tmp_path, monkeypatch, base):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    if base == "master":
        for path in (ctx.settings.canonical_repo, satellite, tmp_path / "canonical"):
            git(path, "branch", "-m", "main", "master")
        git(laptop, "config", "remote.gate.fetch", "+refs/heads/master:refs/remotes/gate/master")
        ctx.settings.default_base_branch = "master"
    old = git(satellite, "rev-parse", "HEAD:formal/model/lib/old.txt").stdout.strip()
    advance(satellite, "formal/model/lib/new.txt")
    new = git(satellite, "rev-parse", "HEAD:formal/model/lib/new.txt").stdout.strip()
    assert not offline_object(laptop, old)
    assert not offline_object(laptop, new)
    before = state(laptop) if base == "main" else git(laptop, "rev-parse", "HEAD").stdout
    with pytest.raises(MirrorError, match="Refusing automatic integration"):
        manager._pull_from_remote(ctx)
    after = state(laptop) if base == "main" else git(laptop, "rev-parse", "HEAD").stdout
    assert after == before
    assert git(laptop, "for-each-ref", "--format=%(objectname)", "refs/sucoder/incoming/").stdout == git(satellite, "rev-parse", "HEAD").stdout
    assert not offline_object(laptop, old)
    assert not offline_object(laptop, new)
    assert not (laptop / "formal/model/lib/new.txt").exists()


def test_filter_unsupported_server_refuses_before_objects_arrive(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    git(satellite, "config", "uploadpack.allowFilter", "false")
    advance(satellite, "formal/model/lib/new.txt")
    before = snapshot(laptop)
    with pytest.raises(MirrorError, match="did not advertise"):
        manager._pull_from_remote(ctx)
    assert snapshot(laptop) == before


def test_dry_run_ignores_a_cached_ahead_tip_and_preserves_all_git_state(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    advance(satellite)
    git(laptop, "fetch", "--filter=blob:none", satellite.as_uri(),
        "+refs/heads/main:refs/sucoder/mirror-head")
    (laptop / "README.md").write_text("unsaved work\n")
    manager.executor.dry_run = True
    before = snapshot(laptop), snapshot(satellite)
    assert manager._pull_from_remote(ctx) is False
    manager.sync(ctx)
    manager.ensure_remote_clone(ctx)
    manager._merge_mirror_commits(str(laptop), "main", "refs/sucoder/mirror-head", "unused")
    manager._stash_mirror_commits(str(laptop), "refs/sucoder/mirror-head", "main")
    manager.bootstrap(ctx)
    manager.launch_agent(ctx)
    assert (snapshot(laptop), snapshot(satellite)) == before


def test_base_only_push_preserves_unrelated_branch_names(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    git(laptop, "checkout", "-qb", "unrelated")
    advance(laptop, "unrelated.txt")
    git(laptop, "checkout", "-q", "main")
    git(satellite, "branch", "unrelated", "main")
    remote_unrelated = git(satellite, "rev-parse", "unrelated").stdout
    advance(laptop, "accepted.txt")
    assert manager._pull_from_remote(ctx)
    manager._sync_remote(ctx)
    assert git(satellite, "rev-parse", "unrelated").stdout == remote_unrelated
    assert git(satellite, "rev-parse", "refs/remotes/ligon/main").stdout == git(laptop, "rev-parse", "main").stdout
    assert git(satellite, "rev-parse", "main").stdout == remote_unrelated


def test_exact_push_lease_refuses_a_concurrent_mirror_writer(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    assert manager._pull_from_remote(ctx)
    manager._sync_remote(ctx)
    advance(laptop, "accepted.txt")
    concurrent = advance(satellite, "concurrent.txt")
    git(satellite, "update-ref", "refs/remotes/ligon/main", concurrent)
    with pytest.raises(MirrorError, match="Failed to push"):
        manager._sync_remote(ctx)
    assert git(satellite, "rev-parse", "main").stdout.strip() == concurrent


def test_reconciliation_refuses_branch_changes_during_the_prompt(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    git(laptop, "config", "remote.gate.promisor", "false")
    advance(satellite, "remote.txt")
    advance(laptop, "local.txt")
    def prompt(*args, **kwargs):
        advance(laptop, "concurrent.txt")
        return "d"
    monkeypatch.setattr(manager, "_prompt_choice", prompt)
    with pytest.raises(MirrorError, match="changed during"):
        manager._pull_from_remote(ctx)
    assert (laptop / "concurrent.txt").exists()


@pytest.mark.parametrize("transport", ["ssh", "https"])
def test_offline_probe_overrides_explicit_transport_permissions(tmp_path, monkeypatch, transport):
    import os
    import shlex
    import subprocess
    from sucoder.remote_bootstrap import VERIFY_LAUNCH_MIRROR_SH
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    missing = advance(satellite, "missing-commit.txt")
    (laptop / ".git/refs/heads/main").write_text(missing + "\n")
    git(laptop, "config", "remote.gate.url", f"{transport}://example.invalid/repo")
    git(laptop, "config", f"protocol.{transport}.allow", "always")
    marker = tmp_path / "transport-invoked"
    if transport == "ssh":
        helper = tmp_path / "ssh-stub"
        helper.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexec git-upload-pack {shlex.quote(str(satellite))}\n")
        helper.chmod(0o700)
        monkeypatch.setenv("GIT_SSH_COMMAND", str(helper))
    else:
        helper = tmp_path / "git-remote-https"
        helper.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexit 1\n")
        helper.chmod(0o700)
        monkeypatch.setenv("GIT_EXEC_PATH", str(tmp_path))
    monkeypatch.delenv("GIT_ALLOW_PROTOCOL", raising=False)
    monkeypatch.delenv("GIT_NO_LAZY_FETCH", raising=False)
    before = snapshot(laptop)
    result = subprocess.run(["bash", "-c", VERIFY_LAUNCH_MIRROR_SH, "probe", str(laptop)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 11
    assert not marker.exists()
    assert snapshot(laptop) == before
    result = manager._canonical_read(str(laptop), "rev-parse", "--verify", "HEAD^{commit}", check=False)
    assert result.returncode != 0
    assert not marker.exists()
    assert snapshot(laptop) == before


@pytest.mark.parametrize("target_kind", ["direct", "confined", "unconfined"])
@pytest.mark.parametrize("command", ["collaborate", "agents-run", "push", "agents-clone"])
def test_remote_cli_dry_run_never_connects_allocates_or_saves(tmp_path, monkeypatch, target_kind, command):
    from sucoder.session import RemoteSession
    from sucoder.mirror import MirrorManager
    path = _write_config(tmp_path, skills_entry=tmp_path / "skills")
    raw = yaml.safe_load(path.read_text())
    if target_kind == "direct":
        target = {"host": "example.invalid"}
    else:
        target = {"gateway": "example.invalid", "transfer_host": "data.invalid",
                  "slurm": {"partition": "test", "account": "test", "cpus_per_task": 1,
                            "mem": "1G", "confined": target_kind == "confined"}}
    raw["targets"] = {"planning": target}
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    def unexpected(*a, **kw):
        pytest.fail("dry run contacted or mutated a remote service")
    for name in ("_connect_with_retry", "_direct_control", "_ensure_slurm_node"):
        monkeypatch.setattr(cli, name, unexpected)
    monkeypatch.setattr(RemoteSession, "save", unexpected)
    monkeypatch.setattr(MirrorManager, "_launch_confined", unexpected)
    monkeypatch.setattr(MirrorManager, "_exec_agent", unexpected)
    args = ["--config", str(path), "-T", "planning", command, "sample", "--dry-run"]
    if command in ("collaborate", "agents-run"):
        args.append("--no-sync")
    if command == "collaborate" and target_kind != "direct":
        args += ["--node", "fake-node"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output


def test_divergent_partial_clone_refuses_automatic_merge_without_hydration(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    advance(satellite, "formal/model/lib/remote.txt")
    oid = git(satellite, "rev-parse", "HEAD:formal/model/lib/remote.txt").stdout.strip()
    (laptop / "local.txt").write_text("distinct local change\n")
    git(laptop, "add", "local.txt")
    git(laptop, "commit", "-qm", "local work")
    assert not offline_object(laptop, oid)
    before = state(laptop)
    with pytest.raises(MirrorError, match="Refusing automatic integration"):
        manager._pull_from_remote(ctx)
    assert state(laptop) == before
    assert not offline_object(laptop, oid)


def test_real_no_sync_launch_preparation_preserves_repositories(tmp_path, monkeypatch):
    from sucoder.executor import CommandResult
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    ctx.settings.sync_policy = "external"
    monkeypatch.setattr(manager, "_compose_context_prelude", lambda *a, **k: "")
    monkeypatch.setattr(manager, "_maybe_run_poetry_auto_install", lambda *a, **k: None)
    monkeypatch.setattr(manager, "_maybe_suggest_mcp_servers", lambda *a, **k: None)
    monkeypatch.setattr(manager, "_record_launch_identity", lambda *a, **k: None)
    run = manager.executor.run_agent
    launched = []
    def stop_at_tmux(args, **kwargs):
        if args[0] == "tmux":
            launched.append(list(args))
            return CommandResult(list(args), list(args), "", "", 0)
        return run(args, **kwargs)
    monkeypatch.setattr(manager.executor, "run_agent", stop_at_tmux)
    before = snapshot(laptop), snapshot(satellite)
    assert manager.bootstrap(ctx) == 0
    assert len([command for command in launched if command[1] == "new-session"]) == 1
    assert (snapshot(laptop), snapshot(satellite)) == before


def test_push_preserves_ignored_remote_collision(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    advance(laptop, "collision.txt")
    (satellite / ".git/info/exclude").write_text("collision.txt\n")
    (satellite / "collision.txt").write_text("irreplaceable remote file\n")
    assert manager._pull_from_remote(ctx)
    before = state(satellite)
    manager._sync_remote(ctx)
    assert state(satellite) == before
    assert git(satellite, "rev-parse", "refs/remotes/ligon/main").stdout == git(laptop, "rev-parse", "main").stdout


def test_publication_refuses_a_sender_rewrite_after_reconciliation(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    old = git(laptop, "rev-parse", "HEAD").stdout.strip()
    reconciled = advance(satellite, "remote-only.txt")
    assert manager._pull_from_remote(ctx)
    manager._sync_remote(ctx)
    git(laptop, "reset", "--hard", old)
    replacement = advance(laptop, "replacement.txt")
    with pytest.raises(MirrorError, match="Failed to push"):
        manager._sync_remote(ctx)
    assert git(satellite, "rev-parse", "HEAD").stdout.strip() == reconciled
    assert git(laptop, "rev-parse", "HEAD").stdout.strip() == replacement


def test_publication_pins_the_accounted_tip_even_if_sender_changes_before_push(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    old = git(laptop, "rev-parse", "HEAD").stdout.strip()
    reconciled = advance(satellite, "remote-only.txt")
    assert manager._pull_from_remote(ctx)
    advance(laptop, "accepted.txt")
    assert manager._pull_from_remote(ctx)
    pinned = git(laptop, "rev-parse", "HEAD").stdout.strip()
    run = manager.executor.run_human
    def rewrite_before_push(args, **kwargs):
        if args[:2] == ["git", "push"]:
            git(laptop, "reset", "--hard", old)
            advance(laptop, "replacement.txt")
        return run(args, **kwargs)
    monkeypatch.setattr(manager.executor, "run_human", rewrite_before_push)
    manager._sync_remote(ctx)
    assert git(satellite, "rev-parse", "refs/remotes/ligon/main").stdout.strip() == pinned
    assert git(satellite, "rev-parse", "HEAD").stdout.strip() == reconciled
    assert git(satellite, "merge-base", "--is-ancestor", reconciled, "refs/remotes/ligon/main").returncode == 0


@pytest.mark.parametrize("command", ["agents-clone", "prepare-canonical", "push", "sync", "pull", "start-task"])
def test_external_policy_cli_refuses_before_executor_construction(tmp_path, monkeypatch, command):
    path = _write_config(tmp_path, skills_entry=tmp_path / "skills")
    raw = yaml.safe_load(path.read_text())
    raw["mirrors"]["sample"]["sync_policy"] = "external"
    raw["targets"] = {"test": {"gateway": "fake.invalid", "transfer_host": "data.invalid",
                                    "slurm": {"partition": "test", "account": "test"}}}
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(cli, "run_startup_checks", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_build_executor", lambda *a, **k: pytest.fail("constructed executor for forbidden operation"))
    args = ["--config", str(path), "-T", "test", command, "sample"]
    if command == "start-task":
        args.append("topic")
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0
    assert "external" in result.output


def test_tracking_publication_preserves_a_file_created_immediately_before_push(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    advance(laptop, "collision.txt")
    (satellite / ".git/info/exclude").write_text("collision.txt\n")
    old = git(satellite, "rev-parse", "HEAD").stdout
    index = (satellite / ".git/index").read_bytes()
    run = manager.executor.run_human
    def inject(args, **kwargs):
        if args[:2] == ["git", "push"]:
            (satellite / "collision.txt").write_text("concurrent irreplaceable file\n")
        return run(args, **kwargs)
    monkeypatch.setattr(manager.executor, "run_human", inject)
    monkeypatch.setattr(manager, "_pull_from_remote", lambda *a: pytest.fail("sync integrated agent work"))
    manager.sync(ctx)
    assert (satellite / "collision.txt").read_text() == "concurrent irreplaceable file\n"
    assert git(satellite, "rev-parse", "HEAD").stdout == old
    assert (satellite / ".git/index").read_bytes() == index
    assert git(satellite, "rev-parse", "refs/remotes/ligon/main").stdout == git(laptop, "rev-parse", "HEAD").stdout


def test_tracking_publication_preserves_a_concurrent_agent_commit(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    source = advance(laptop, "accepted.txt")
    run = manager.executor.run_human
    agent = []
    def inject(args, **kwargs):
        if args[:2] == ["git", "push"]:
            agent.append(advance(satellite, "agent-only.txt"))
        return run(args, **kwargs)
    monkeypatch.setattr(manager.executor, "run_human", inject)
    manager.sync(ctx)
    assert git(satellite, "rev-parse", "HEAD").stdout.strip() == agent[0]
    assert git(satellite, "rev-parse", "refs/remotes/ligon/main").stdout.strip() == source
    assert (satellite / "agent-only.txt").exists()
    assert not (laptop / "agent-only.txt").exists()


def test_existing_dirty_mirror_publication_never_rescues_stashes_or_resets(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    advance(laptop, "accepted.txt")
    (satellite / "README.md").write_text("unsaved agent edit\n")
    (satellite / "untracked.txt").write_text("untracked agent work\n")
    git(satellite, "add", "README.md")
    before = state(satellite), (satellite / ".git/config").read_bytes()
    monkeypatch.setattr(manager, "_prompt_choice", lambda *a, **k: pytest.fail("publication prompted about checkout"))
    assert manager.ensure_remote_clone(ctx)
    assert (state(satellite), (satellite / ".git/config").read_bytes()) == before


def test_initial_checkout_refuses_a_late_ignored_collision(tmp_path, monkeypatch):
    from tests.test_remote import _local_bootstrap_manager
    manager, ctx, path = _local_bootstrap_manager(tmp_path, monkeypatch)
    run = manager.executor.run_agent
    def inject(args, **kwargs):
        if args[:2] == ["git", "checkout"]:
            (path / ".git/info/exclude").write_text("README.md\n")
            (path / "README.md").write_text("concurrent initial work\n")
        return run(args, **kwargs)
    monkeypatch.setattr(manager.executor, "run_agent", inject)
    with pytest.raises(MirrorError, match="Initial mirror checkout refused"):
        manager.ensure_remote_clone(ctx)
    assert (path / "README.md").read_text() == "concurrent initial work\n"
    assert git(path, "rev-parse", "--verify", "HEAD", check=False).returncode != 0


def test_remote_task_uses_published_baseline_without_rewriting_human_alias(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    old = git(satellite, "rev-parse", "HEAD").stdout
    git(satellite, "branch", "ligon/main", "HEAD")
    accepted = advance(laptop, "accepted.txt")
    monkeypatch.setattr(manager, "_ensure_mirror_exists", lambda c: satellite)
    task = manager.start_task(ctx, task_name="review", timestamp="20261025")
    assert git(satellite, "rev-parse", "HEAD").stdout.strip() == accepted
    assert git(satellite, "rev-parse", "ligon/main").stdout == old
    assert git(satellite, "rev-parse", "main").stdout == old
    assert git(satellite, "symbolic-ref", "--short", "HEAD").stdout.strip() == task
    agent = advance(satellite, "agent-work.txt")
    with pytest.raises(MirrorError, match="Task branch creation refused"):
        manager.start_task(ctx, task_name="review", timestamp="20261025")
    assert git(satellite, "rev-parse", "HEAD").stdout.strip() == agent
    assert (satellite / "agent-work.txt").exists()


def test_explicit_full_pull_protects_a_file_created_after_preflight(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    advance(satellite, "collision.txt")
    (laptop / ".git/info/exclude").write_text("collision.txt\n")
    old = git(laptop, "rev-parse", "HEAD").stdout
    run = manager.executor.run_human
    def inject(args, **kwargs):
        if args[:2] == ["git", "merge"]:
            (laptop / "collision.txt").write_text("concurrent local work\n")
        return run(args, **kwargs)
    monkeypatch.setattr(manager.executor, "run_human", inject)
    with pytest.raises(MirrorError, match="Cannot safely fast-forward"):
        manager._pull_from_remote(ctx)
    assert (laptop / "collision.txt").read_text() == "concurrent local work\n"
    assert git(laptop, "rev-parse", "HEAD").stdout == old


def test_partial_publication_refuses_before_any_transfer(tmp_path, monkeypatch):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    before = snapshot(laptop), snapshot(satellite)
    monkeypatch.setattr(manager, "_git_transports", lambda *a: pytest.fail("prepared transfer from partial follower"))
    with pytest.raises(MirrorError, match="partial clone"):
        manager.sync(ctx)
    assert (snapshot(laptop), snapshot(satellite)) == before


def test_fresh_local_tier_mailbox_still_accepts_agent_publication(tmp_path, monkeypatch):
    from sucoder.config import SlurmConfig
    from sucoder.local_tier import build_prepare_script
    from tests.test_remote import _local_bootstrap_manager
    import subprocess
    manager, ctx, path = _local_bootstrap_manager(tmp_path, monkeypatch)
    ctx.settings.remote.slurm = SlurmConfig(partition="test", account="test", local_disk="/local")
    assert manager.ensure_remote_clone(ctx)
    assert git(path, "config", "receive.denyCurrentBranch").stdout.strip() == "updateInstead"
    root = tmp_path / "node-local"
    script = build_prepare_script(mirror_path=str(path), mirror_token="rproj",
                                  local_disk_root=str(root), job_id=7, canonical_remote="ligon")
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    work = root / "job7/mirrors/rproj"
    agent = advance(work, "agent.txt")
    assert git(path, "rev-parse", "main").stdout.strip() == agent
    assert (path / "agent.txt").exists()


@pytest.mark.parametrize("policy, expected", [("canonical", "refs/remotes/ligon/"), ("external", "external integration workflow")])
def test_agent_context_reports_the_actual_authority_without_probing_a_base(tmp_path, monkeypatch, policy, expected):
    manager, ctx, laptop, satellite = filtered_topology(tmp_path, monkeypatch)
    ctx.settings.sync_policy = policy
    for name in ("_system_prompt_block", "_target_prompt_block", "_workspace_block", "_agent_doc_block"):
        monkeypatch.setattr(manager, name, lambda *a: None)
    monkeypatch.setattr(manager, "_skill_blocks", lambda *a: [])
    monkeypatch.setattr(manager, "_warn_if_skills_base_unusable", lambda *a: None)
    monkeypatch.setattr(manager, "_resolve_base_branch", lambda *a: pytest.fail("probed a base during context rendering"))
    before = snapshot(laptop), snapshot(satellite)
    assert expected in manager._compose_context_prelude(ctx)
    assert (snapshot(laptop), snapshot(satellite)) == before


@pytest.mark.parametrize("marker", ["MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply", "sequencer"])
def test_explicit_pull_preserves_a_pending_operation_even_with_a_clean_index(tmp_path, monkeypatch, marker):
    manager, ctx, laptop, satellite = full_topology(tmp_path, monkeypatch)
    advance(satellite, "incoming.txt")
    path = laptop / ".git" / marker
    if marker.endswith("HEAD"):
        path.write_text(git(laptop, "rev-parse", "HEAD").stdout)
    else:
        path.mkdir()
        (path / "saved-operation").write_text("preserve state\n")
    before = state(laptop), snapshot(path) if path.is_dir() else path.read_bytes()
    monkeypatch.setattr(manager, "_prompt_choice", lambda *a, **k: pytest.fail("prompted during a pending operation"))
    with pytest.raises(MirrorError, match="already in progress"):
        manager._pull_from_remote(ctx)
    assert (state(laptop), snapshot(path) if path.is_dir() else path.read_bytes()) == before
