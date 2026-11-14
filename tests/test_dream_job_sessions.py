"""Historical clone discovery needs configured layout and native repo metadata."""
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import pytest

from sucoder import dream, dream_experience, dream_sessions
from sucoder.config import MirrorSettings, BranchPrefixes
from sucoder.local_tier import work_path
from tests.test_dream import workspace  # noqa: F401
from tests.test_dream_local import local  # noqa: F401


def session(home, cwd, number, *, metadata_cwd=None, branch="coder/old-work", age=3600):
    sid = f"00000000-0000-4000-8000-{number:012d}"
    directory = dream_experience.project_dir(home, cwd)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (sid + ".jsonl")
    row = dict(type="user", sessionId=sid, cwd=metadata_cwd or str(cwd), gitBranch=branch,
        message=dict(role="user", content="A recorded local-tier lesson."))
    path.write_text(json.dumps(row) + "\n")
    os.utime(path, (time.time() - age, time.time() - age))
    return path, sid


def source(config, clone):
    item = replace(config.mirrors["sample"], canonical_repo=clone, mirror_name=clone.name)
    bound = replace(config, mirror_root=clone.parent, mirrors={"sample": item}, targets={})
    return dream.GitSource(bound, "sample", time.monotonic() + 30)


def test_earlier_job_sessions_are_found_after_the_clone_disappeared(workspace, local):
    clone, config = local
    home = Path(os.environ["HOME"])
    disk = config.targets["cluster"].slurm.local_disk
    old = Path(work_path(disk, "sample", 3))
    transcript, sid = session(home, old, 1)
    assert not old.exists()
    rows = dream_experience.sessions(source(config, clone), config=config, home=home)
    assert [(row["thread_id"], row["workspace"], row["state"]) for row in rows] == [(sid, str(old), "finished")]
    assert rows[0]["path"] == str(transcript)
    assert dream_experience.sessions(source(config, clone), home=home) == []


def test_same_repository_can_discover_old_job_sessions_from_the_durable_mirror(workspace, local):
    clone, config = local
    home = Path(os.environ["HOME"])
    old = Path(work_path(config.targets["cluster"].slurm.local_disk, "sample", 8))
    _, sid = session(home, old, 2)
    shared = dream.GitSource(config, "sample", time.monotonic() + 30)
    assert [row["thread_id"] for row in dream_experience.sessions(shared, config=config, home=home)] == [sid]


@pytest.mark.parametrize("defect", ["cwd-collision", "missing-branch", "wrong-session"])
def test_lossy_project_slug_is_not_repository_identity(workspace, local, defect):
    clone, config = local
    home = Path(os.environ["HOME"])
    old = Path(work_path(config.targets["cluster"].slurm.local_disk, "sample", 4))
    path, sid = session(home, old, 3,
        metadata_cwd=str(old).replace("/mirrors/", "-mirrors/") if defect == "cwd-collision" else None,
        branch="" if defect == "missing-branch" else "main")
    if defect == "wrong-session":
        row = json.loads(path.read_text())
        row["sessionId"] = "another-session"
        path.write_text(json.dumps(row) + "\n")
        os.utime(path, (time.time() - 3600, time.time() - 3600))
    assert dream_experience.sessions(source(config, clone), config=config, home=home) == []


def test_other_project_directories_are_never_opened(workspace, local, monkeypatch):
    clone, config = local
    home = Path(os.environ["HOME"])
    disk = config.targets["cluster"].slurm.local_disk
    session(home, work_path(disk, "sample", 7), 4)
    alien, _ = session(home, work_path(disk, "another-project", 7), 5)
    original = dream_sessions.dream_transcript._open_source
    opened = []
    def read(path):
        opened.append(path)
        assert path != alien
        return original(path)
    monkeypatch.setattr(dream_sessions.dream_transcript, "_open_source", read)
    assert len(dream_experience.sessions(source(config, clone), config=config, home=home)) == 1
    assert opened and alien not in opened


def test_historical_discovery_preserves_live_and_quiet_exclusions(workspace, local):
    clone, config = local
    home = Path(os.environ["HOME"])
    old = work_path(config.targets["cluster"].slurm.local_disk, "sample", 9)
    _, finished = session(home, old, 6)
    _, recent = session(home, old, 7, age=0)
    _, live = session(home, old, 8)
    rows = dream_experience.sessions(source(config, clone), config=config, home=home, live={live})
    assert {row["thread_id"]: row["state"] for row in rows} == {finished: "finished", recent: "recent", live: "live"}


def test_ambiguous_sanitized_mirror_tokens_do_not_admit_old_sessions(workspace, local):
    clone, config = local
    home = Path(os.environ["HOME"])
    shared = dream.GitSource(config, "sample", time.monotonic() + 30)
    session(home, work_path(config.targets["cluster"].slurm.local_disk, "sample_", 10), 9)
    other = MirrorSettings(name="sample?", canonical_repo=clone.parent / "unrelated", mirror_name="unrelated",
        branch_prefixes=BranchPrefixes())
    config = replace(config, mirrors={"sample!": replace(config.mirrors["sample"], name="sample!"), "sample?": other})
    assert dream_experience.sessions(shared, config=config, home=home) == []
