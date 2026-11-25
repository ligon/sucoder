"""Project links from a read-only copy of the memory wiki (essay-alignment S5)."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from sucoder import dream, dream_explore as explore, dream_memory, dream_reader, dream_run, dream_wiki as wiki
from tests.test_dream import git, workspace  # noqa: F401
from tests.test_dream_explore import ExploringTransport, clock, options  # noqa: F401
from tests.test_dream_explore_cli import invoke
from tests.test_dream_inject import pages
from tests.test_dream_wiki import EditorTransport, TOPIC, planned

MEMORY = "refs/heads/sucoder/memory"
LINKED = {"index.org": "- [[file:topics/notes.org][Notes]]\n",
          "topics/notes.org": "* Notes\nSee [[file:../../../note.org][the project notes]].\n"}


def copy_of(repo):
    return repo / ".sucoder" / "wiki"


def test_the_copy_is_written_read_only_marked_excluded_and_its_links_resolve(workspace):
    repo = workspace[0]
    commit = pages(repo, LINKED)
    result = dream_memory.materialize(repo, "sucoder/memory")
    assert result == dict(state="written", ref=MEMORY, commit=commit, path=str(copy_of(repo)), files=2)
    page = copy_of(repo) / "topics/notes.org"
    assert page.read_text() == LINKED["topics/notes.org"] and not page.stat().st_mode & 0o222
    assert (page.parent / "../../../note.org").resolve() == (repo / "note.org").resolve()
    marker = json.loads((copy_of(repo) / dream_memory.MARKER).read_text())
    assert marker["commit"] == commit and marker["ref"] == MEMORY
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    exclude = (repo / ".git/info/exclude").read_text().splitlines()
    assert all(exclude.count(line) == 1 for line in dream_memory.EXCLUDES)
    assert dream_memory.materialize(repo, "sucoder/memory")["state"] == "current"
    assert all(exclude.count(line) == 1 for line in (repo / ".git/info/exclude").read_text().splitlines()
               if line in dream_memory.EXCLUDES)


def test_a_new_generation_replaces_the_copy_and_leaves_no_staging(workspace):
    repo = workspace[0]
    pages(repo, LINKED)
    dream_memory.materialize(repo, "sucoder/memory")
    newer = pages(repo, {"index.org": "- only the index now\n"})
    assert dream_memory.materialize(repo, "sucoder/memory")["commit"] == newer
    assert not (copy_of(repo) / "topics").exists() and "only the index" in (copy_of(repo) / "index.org").read_text()
    assert [p.name for p in (repo / ".sucoder").iterdir() if p.name.startswith(".wiki-") and p.name != ".wiki-lock"] == []


def test_an_absent_branch_reports_absent(workspace):
    assert dream_memory.materialize(workspace[0], "sucoder/memory") == dict(state="absent", ref=MEMORY)


def test_a_tracked_or_foreign_wiki_is_left_alone(workspace, tmp_path):
    repo = workspace[0]
    pages(repo, LINKED)
    foreign = copy_of(repo)
    foreign.mkdir(parents=True)
    (foreign / "mine.org").write_text("a human's own notes\n")
    assert dream_memory.materialize(repo, "sucoder/memory")["state"] == "unmanaged"
    assert (foreign / "mine.org").read_text() == "a human's own notes\n"
    git(repo, "add", ".sucoder/wiki/mine.org")
    assert dream_memory.materialize(repo, "sucoder/memory")["state"] == "tracked"
    git(repo, "rm", "-q", "--cached", ".sucoder/wiki/mine.org")
    (foreign / "mine.org").unlink()
    foreign.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    foreign.symlink_to(elsewhere)
    assert dream_memory.materialize(repo, "sucoder/memory")["state"] == "unmanaged"
    assert list(elsewhere.iterdir()) == []


def test_a_held_lock_reports_busy(workspace):
    repo = workspace[0]
    pages(repo, LINKED)
    (repo / ".sucoder").mkdir()
    with open(repo / ".sucoder/.wiki-lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert dream_memory.materialize(repo, "sucoder/memory")["state"] == "busy"
    assert dream_memory.materialize(repo, "sucoder/memory")["state"] == "written"


def test_a_non_regular_entry_or_a_subdirectory_is_refused(workspace):
    repo = workspace[0]
    env = dict(os.environ, GIT_INDEX_FILE=str(repo.parent / "odd-index"))
    blob = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input=b"../../etc",
                          capture_output=True, check=True).stdout.decode().strip()
    subprocess.run(["git", "-C", str(repo), "read-tree", "--empty"], env=env, check=True)
    subprocess.run(["git", "-C", str(repo), "update-index", "--add", "--cacheinfo",
                    "120000," + blob + ",.sucoder/wiki/escape"], env=env, check=True)
    tree = subprocess.run(["git", "-C", str(repo), "write-tree"], env=env, capture_output=True, check=True).stdout.decode().strip()
    git(repo, "update-ref", MEMORY, git(repo, "commit-tree", tree, "-m", "odd"))
    with pytest.raises(dream.DreamError, match="non-regular"):
        dream_memory.materialize(repo, "sucoder/memory")
    assert not copy_of(repo).exists() or not (copy_of(repo) / "escape").exists()
    (repo / "sub").mkdir()
    with pytest.raises(dream.DreamError, match="root of a git work tree"):
        dream_memory.materialize(repo / "sub", "sucoder/memory")


def test_the_launcher_entry_point_prints_its_state(workspace):
    repo = workspace[0]
    pages(repo, LINKED)
    done = subprocess.run([os.sys.executable, "-m", "sucoder.dream_memory", "checkout", str(repo), "sucoder/memory"],
                          capture_output=True, text=True, cwd="/", env=dict(os.environ, PYTHONPATH=str(Path.cwd())))
    assert done.returncode == 0 and json.loads(done.stdout)["state"] == "written"


def test_dream_checkout_writes_the_copy_for_any_checkout(workspace):
    repo = workspace[0]
    pages(repo, LINKED)
    result = invoke(workspace, "checkout", str(repo))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["state"] == "written" and (copy_of(repo) / "index.org").exists()


def test_plans_record_the_projects_regular_files_and_replay_them(workspace):
    plan = planned(workspace)
    repo, _, config, request = workspace
    (repo / "link").symlink_to("note.org")
    git(repo, "add", "link")
    git(repo, "commit", "-qm", "a symlink is not linkable")
    catalog = Path(workspace[1]).parent / "linked-sources.json"
    catalog.write_text(json.dumps(dict(request, units=[dict(unit_id="a", kind="git", shareable=True,
        locator=dict(repository="sample", commit="HEAD", path="evidence-a.org")), dict(unit_id="b", kind="git",
        shareable=True, locator=dict(repository="sample", commit="HEAD", path="evidence-b.org"))])))
    linked = dream.create_plan(config, "sample", catalog, "refs/heads/main", project_commit="HEAD")
    assert linked["project"] == dict(commit=git(repo, "rev-parse", "HEAD"),
                                     paths=["evidence-a.org", "evidence-b.org", "note.org"])
    assert "project" not in plan
    assert dream_run.snapshot(config, "sample", Path(linked["plan_path"]))["state"] == "snapshotted"


def proposal(link, *, page=".sucoder/wiki/topics/rule.org"):
    topic = "* Rule\n:PROPERTIES:\n:CUSTOM_ID: rule\n:END:\nExplanation; see [[file:" + link + "][a file]].\n"
    return dict(pages=[dict(path=wiki.INDEX, content="[[file:topics/rule.org::#rule][Rule]]\n"), dict(path=page, content=topic)],
                changes=[dict(path=p, before_sha256="", reason="Explain") for p in (wiki.INDEX, page)],
                references=[dict(path=page, target="rule", evidence_status="inference",
                                 citations=[dict(unit_id="selected", quote="Admitted evidence.")])], omissions=[])


SOURCE = dict(unit_id="selected", repository="sample", path="docs/admitted.org", byte_start=0, byte_end=19,
              text="Admitted evidence.\n")


@pytest.mark.parametrize("link, project, error", [
    ("../../../sucoder/attention.py", ["sucoder/attention.py"], None),
    ("../../../docs/admitted.org", [], None),
    ("../../../sucoder/missing.py", ["sucoder/attention.py"], "outside the admitted source set and the project: sucoder/missing.py"),
    ("../../../sucoder/attention.py", None, "outside the admitted source set: sucoder/attention.py"),
    ("../../../../sucoder/attention.py", ["sucoder/attention.py"], "climbs above the project root"),
])
def test_links_may_name_any_project_file_but_nothing_else(link, project, error):
    plan = dict(repository="sample", limits=dict(max_consolidation_bytes=65536))
    if project is not None:
        plan["project"] = dict(commit="c" * 40, paths=project)
    if error is None:
        wiki.validate_proposal(plan, proposal(link), {}, [SOURCE], time.monotonic() + 5)
    else:
        with pytest.raises(dream.DreamError, match=error):
            wiki.validate_proposal(plan, proposal(link), {}, [SOURCE], time.monotonic() + 5)


def test_a_dream_links_an_unadmitted_project_file_and_it_resolves_in_the_copy(workspace, clock, monkeypatch):
    monkeypatch.setattr(explore, "_live_sessions", lambda config: set())
    planned(workspace)
    repo, _, config, _ = workspace
    dream_memory.ensure(dream.GitSource(config, "sample", time.monotonic() + 30), MEMORY,
                        git(repo, "rev-parse", "HEAD"), repo.parent, time.monotonic() + 30)
    git(repo, "rm", "-rq", ".sucoder/wiki")  # The memory branch is the copy of record (S5).
    git(repo, "commit", "-qm", "Keep the wiki on its memory branch only")

    def link(task, result):
        if task["role"] in ("author", "repair"):
            page = next(p for p in result["pages"] if p["path"] == TOPIC)
            page["content"] = page["content"].replace(
                "precede the next input.\n", "precede the next input.\nLaunch notes: [[file:../../../note.org][project notes]].\n")
    native = ExploringTransport(editor=EditorTransport(mutate=link))
    result = explore.start(config, cwd=repo, overrides=options(mode="project"), transport=native)
    assert result["state"] == "published"
    plan = json.loads(Path(result["plan_path"]).read_bytes())
    assert "note.org" in plan["project"]["paths"] and not any(p.startswith(".sucoder/") for p in plan["project"]["paths"])
    page = copy_of(repo) / "topics/reply-evidence.org"
    assert "project notes" in page.read_text() and not page.stat().st_mode & 0o222
    assert (page.parent / "../../../note.org").resolve() == (repo / "note.org").resolve()
    marker = json.loads((copy_of(repo) / dream_memory.MARKER).read_text())
    assert marker["commit"] == git(repo, "rev-parse", MEMORY)
    assert git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    read = dream_reader.read(config, "sample", topic="reply-evidence", base=None)
    assert dict(link="../../../note.org", path="note.org", present_at_head=True) in read["links"]
