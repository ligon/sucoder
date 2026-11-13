"""sucoder – Unix-sandboxed agent collaboration toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sucoder")
except PackageNotFoundError:
    __version__ = "0.0.0"

# The oldest interpreter SuCoder's staged helpers are written for, on the
# controller (pyproject's python = "^3.9") and on every target.
PYTHON_FLOOR = (3, 9)

__all__ = ["PYTHON_FLOOR", "__version__"]
