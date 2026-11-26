"""Python children import the parent's own sucoder (sucoder/child_python.py)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import sucoder
from sucoder import child_python

PACKAGE = str(Path(sucoder.__file__).resolve().parent)
REPORT = "import json,sys,sucoder; print(json.dumps(dict(package=sucoder.__file__, argv=sys.argv)))"


def run(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, timeout=30, **kwargs)


def test_the_root_is_the_directory_this_process_imported_sucoder_from():
    assert os.path.join(child_python.ROOT, "sucoder") == PACKAGE


def test_a_module_child_runs_this_package_over_another_on_its_path(foreign_sucoder):
    plain = run([sys.executable, "-m", "sucoder.dream_memory"], cwd="/")
    assert plain.returncode == 1 and "foreign sucoder imported" in plain.stderr  # The defect.
    pinned = run(child_python.module_argv("sucoder.dream_memory"), cwd="/")
    # This package's dream_memory ran as __main__: usage on bad arguments, exit 2.
    assert pinned.returncode == 2, pinned.stderr
    assert "usage: python -m sucoder.dream_memory checkout PATH BRANCH" in pinned.stderr


def test_a_child_keeps_its_arguments(foreign_sucoder):
    done = run(child_python.argv(REPORT, "--job", "x y", ""), cwd="/")
    assert done.returncode == 0, done.stderr
    value = json.loads(done.stdout)
    assert str(Path(value["package"]).resolve().parent) == PACKAGE
    assert value["argv"] == ["-c", "--job", "x y", ""]


def test_a_copy_in_the_working_directory_does_not_shadow_this_package(tmp_path, monkeypatch):
    # e.g. `sucoder dream --detach` run inside a checkout of another SuCoder version.
    project = tmp_path / "project"
    (project / "sucoder").mkdir(parents=True)
    (project / "sucoder" / "__init__.py").write_text('raise ImportError("working-directory sucoder imported")\n')
    monkeypatch.setenv("PYTHONPATH", child_python.ROOT)
    plain = run([sys.executable, "-c", REPORT], cwd=project)
    assert "working-directory sucoder imported" in plain.stderr  # The defect.
    pinned = run(child_python.argv(REPORT), cwd=project)
    assert pinned.returncode == 0, pinned.stderr
    assert str(Path(json.loads(pinned.stdout)["package"]).resolve().parent) == PACKAGE


def test_a_child_that_already_finds_this_package_keeps_its_path(monkeypatch):
    # As for an installed package: the pin is a no-op, sys.path is untouched.
    monkeypatch.setenv("PYTHONPATH", child_python.ROOT)
    code = "import json,sys; print(json.dumps(sys.path))"
    plain = run([sys.executable, "-c", code], cwd="/")
    pinned = run(child_python.argv(code), cwd="/")
    assert plain.returncode == pinned.returncode == 0, plain.stderr + pinned.stderr
    assert json.loads(pinned.stdout) == json.loads(plain.stdout)
