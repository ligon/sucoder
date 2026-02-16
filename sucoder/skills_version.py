"""Skills repository version validation.

This module enforces version compatibility between the sucoder tool
and the skills repository using semantic versioning.
"""

from pathlib import Path
from packaging import version


# Required skills repository version
# Tool is compatible with skills versions in the same major version.
# Note: the tool version (e.g. 0.1.0 in pyproject.toml) and the skills
# version (1.0.0) are intentionally independent — they follow separate
# release cadences and the tool only constrains the skills major version.
REQUIRED_SKILLS_VERSION = "1.0.0"


class SkillsVersionError(Exception):
    """Raised when skills version is incompatible with tool."""
    pass


def validate_skills_version(skills_dir: Path) -> None:
    """Validate that skills repository version is compatible with this tool.

    Args:
        skills_dir: Path to skills repository root directory

    Raises:
        SkillsVersionError: If VERSION file is missing or incompatible

    Version Compatibility Rules:
        - Major version must match (1.x.x compatible with 1.y.z)
        - Skills version must be >= minimum required version
        - Skills version must be < next major version (breaking changes)

    Examples:
        Tool requires 1.0.0:
        - Skills 1.2.5 → Compatible ✓
        - Skills 0.9.0 → Incompatible (too old) ✗
        - Skills 2.0.0 → Incompatible (breaking changes) ✗
    """
    version_file = skills_dir / "VERSION"

    # Check if VERSION file exists
    if not version_file.exists():
        raise SkillsVersionError(
            f"Skills repository missing VERSION file.\n"
            f"Expected VERSION file at: {version_file}\n"
            f"Required version: {REQUIRED_SKILLS_VERSION}\n\n"
            f"The skills repository must include a VERSION file for compatibility checking.\n"
            f"If you just updated skills, ensure VERSION file was included."
        )

    # Read and parse actual version
    try:
        actual_version_str = version_file.read_text().strip()
        actual = version.parse(actual_version_str)
    except Exception as e:
        raise SkillsVersionError(
            f"Failed to parse VERSION file at: {version_file}\n"
            f"Error: {e}\n\n"
            f"VERSION file should contain a single semantic version like: 1.0.0"
        )

    # Parse required version
    required = version.parse(REQUIRED_SKILLS_VERSION)

    # Calculate maximum compatible version (next major version)
    max_major = required.major + 1
    max_version = version.parse(f"{max_major}.0.0")

    # Check major version compatibility
    if actual.major != required.major:
        if actual.major < required.major:
            raise SkillsVersionError(
                f"Skills version is too old.\n"
                f"  Skills version: {actual}\n"
                f"  Required version: {REQUIRED_SKILLS_VERSION} or higher\n"
                f"  Max compatible: < {max_version}\n\n"
                f"Your skills repository is out of date and missing required features.\n\n"
                f"To update:\n"
                f"  cd {skills_dir}\n"
                f"  git pull\n\n"
                f"Or to skip version check (not recommended):\n"
                f"  export SUCODER_SKIP_SKILLS_VERSION=1"
            )
        else:
            raise SkillsVersionError(
                f"Skills version is too new (breaking changes).\n"
                f"  Skills version: {actual}\n"
                f"  Required version: {REQUIRED_SKILLS_VERSION}\n"
                f"  Max compatible: < {max_version}\n\n"
                f"Your skills repository has breaking changes not supported by this tool version.\n\n"
                f"To fix, either:\n"
                f"1. Upgrade sucoder:\n"
                f"     pip install --upgrade sucoder\n"
                f"2. Downgrade skills to v{required.major}.x:\n"
                f"     cd {skills_dir}\n"
                f"     git checkout v{required.major}.9.9  # Or latest {required.major}.x version\n\n"
                f"Or to skip version check (not recommended):\n"
                f"  export SUCODER_SKIP_SKILLS_VERSION=1"
            )

    # Check minimum version within major version
    if actual < required:
        raise SkillsVersionError(
            f"Skills version is too old.\n"
            f"  Skills version: {actual}\n"
            f"  Required version: {REQUIRED_SKILLS_VERSION} or higher\n"
            f"  Max compatible: < {max_version}\n\n"
            f"Your skills repository is missing features required by this tool version.\n\n"
            f"To update:\n"
            f"  cd {skills_dir}\n"
            f"  git pull\n\n"
            f"Or to skip version check (not recommended):\n"
            f"  export SUCODER_SKIP_SKILLS_VERSION=1"
        )

    # If we get here, version is compatible
    # Optionally log success (can add logger.debug here if needed)
