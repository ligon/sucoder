"""Group permissions on agent-owned mirrors (sucoder/permissions.py)."""
import getpass
import grp
import logging
import os
import subprocess
from types import SimpleNamespace

from sucoder.permissions import apply_agent_repo_permissions


class LocalExecutor:
    """Runs the agent's commands as this account, which owns the test tree."""

    agent_user = getpass.getuser()
    logger = logging.getLogger("sucoder.test")

    def __init__(self):
        self.calls = []

    def run_agent(self, args, check=False, **kwargs):
        self.calls.append(list(args))
        done = subprocess.run(list(args), capture_output=True, text=True, check=False)
        return SimpleNamespace(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr)


def test_group_write_skips_the_read_only_memory_copy(tmp_path):
    # A path holding find pattern characters must still name only itself.
    repo = tmp_path / "mirror [*?] name"
    wiki = repo / ".sucoder" / "wiki"
    (wiki / "topics").mkdir(parents=True)
    page, index, code = wiki / "topics" / "notes.org", wiki / "index.org", repo / "code.py"
    for path in (page, index, code):
        path.write_text("x\n")
    for path, mode in ((page, 0o440), (index, 0o440), (wiki / "topics", 0o750), (wiki, 0o750), (code, 0o640)):
        path.chmod(mode)
    group = grp.getgrgid(os.getgid()).gr_name
    executor = LocalExecutor()
    apply_agent_repo_permissions(executor, repo, agent_group=group)
    assert code.stat().st_mode & 0o777 == 0o660
    assert (repo / ".sucoder").stat().st_mode & 0o070 == 0o070
    for path in (page, index):
        assert path.stat().st_mode & 0o777 == 0o440
    for path in (wiki, wiki / "topics"):
        assert path.stat().st_mode & 0o777 == 0o750 and path.stat().st_mode & 0o2000
    chmod = next(call for call in executor.calls if "g+rwX" in call)
    assert chmod[1:5] == [str(repo), "-path", str(wiki).replace("[", "\\[").replace("]", "\\]")
                          .replace("*", "\\*").replace("?", "\\?"), "-prune"]
