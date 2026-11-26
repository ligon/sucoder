"""Start a Python child that imports this process's own ``sucoder``.

``python -m sucoder.NAME`` imports whichever ``sucoder`` the child finds on
its own path.  That is this process's package only when the interpreter has
it installed.  A source checkout reached through ``PYTHONPATH`` (which
``sudo`` drops), through the parent's working directory or through a test
runner is missing from the child, which then cannot start; a ``sucoder`` in
the child's working directory, or an older installed copy, is found instead,
so the child runs code its parent did not.  The launch's read-only memory
copy was lost silently this way.

:func:`argv` hands the child this package's root and puts it first on
``sys.path`` only when the child would otherwise find another package or
none, so a child of an installed package runs exactly as before.  It changes
no environment variable: a ``sudo`` or sandboxed child keeps the environment
its caller chose.
"""
from __future__ import annotations

import os
import sys
from typing import List

# The directory holding the package this process imported.
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# ``python -c _PIN+CODE ROOT ARGS...``: _PIN takes ROOT back out of sys.argv,
# so CODE sees the argv of ``python -c CODE ARGS...``.  find_spec locates the
# child's own ``sucoder`` without importing it, so a different one never runs.
_PIN = ("import os,sys; from importlib.util import find_spec; r=sys.argv.pop(1); s=find_spec('sucoder'); "
        "(s and s.origin and os.path.dirname(os.path.dirname(os.path.realpath(s.origin))))==r "
        "or sys.path.insert(0,r); del r,s,find_spec; ")


def argv(code: str, *args: str) -> List[str]:
    """Like ``[python, "-c", code, *args]``, importing this process's ``sucoder``."""
    return [sys.executable, "-c", _PIN + code, ROOT, *args]


def module_argv(module: str, *args: str) -> List[str]:
    """Like ``[python, "-m", module, *args]``, importing this process's ``sucoder``."""
    return argv("import runpy; runpy.run_module(%r, run_name='__main__', alter_sys=True)" % module, *args)
