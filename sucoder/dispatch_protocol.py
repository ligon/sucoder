"""Shared headless dispatch constants; Python 3.6 and standard library only."""

import json
import os
import tempfile

SCHEMA_VERSION = 1
TERMINAL_STATES = frozenset(("succeeded", "failed", "refused", "interrupted"))

# These are provider settings or credentials, never coordinator runtime state.
# The controller stores values privately; the receiving worker permits exactly
# the same keys. Keep this file staged alongside the standalone worker.
ENVIRONMENT_ALLOWLIST = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID",
    "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "AWS_DEFAULT_REGION", "ANTHROPIC_VERTEX_PROJECT_ID",
    "CLOUD_ML_REGION", "GOOGLE_CLOUD_PROJECT",
)


def atomic_json(path, value):
    """Publish a private receipt with atomic replacement after a complete write."""
    fd, temporary = tempfile.mkstemp(prefix=".dispatch-", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
