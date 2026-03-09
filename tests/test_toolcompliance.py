"""
Test the code's compliance with our chosen tooling.

Did the developer run all the tools, and fix all the errors that were reported.
"""

import pathlib
import subprocess

import pytest
from mypy import api
from ruff.__main__ import find_ruff_bin  # type: ignore[import-untyped]

from tests import IN_GITHUB_ACTIONS

# Ignore import untyped since no type stub available for ruff.__main__

RUFF_ASSERT_CODE = 1
RUFF_ERROR_CODE = 2


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipped in CI")
def test_ruff_format() -> None:
    """Whether ruff format would reformat a file."""
    ruff: str = find_ruff_bin()
    command = [ruff, "format", "--quiet", "--check", "--force-exclude"]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)  # noqa: S603
    # No S603 since we are pretty certain the input is not dangerous
    stdout = completed.stdout
    stderr = completed.stderr

    if completed.returncode == RUFF_ASSERT_CODE:
        msg = "Ruff format: at least one file would be reformatted.\n" + stdout
        raise AssertionError(msg)

    if completed.returncode == RUFF_ERROR_CODE:
        raise RuntimeError("Ruff format encountered an error.\n" + stderr)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipped in CI")
def test_ruff_linting() -> None:
    """Whether ruff check would report any issues."""
    ruff: str = find_ruff_bin()
    command = [ruff, "check", "--quiet", "--output-format=full", "--force-exclude"]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)  # noqa: S603
    # No S603 since we are pretty certain the input is not dangerous
    stdout = completed.stdout
    stderr = completed.stderr

    if completed.returncode == RUFF_ASSERT_CODE:
        raise AssertionError("Ruff check found at least one issue.\n" + stdout)
    if completed.returncode == RUFF_ERROR_CODE:
        raise RuntimeError("Ruff check encountered an error.\n" + stderr)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skipped in CI")
def test_mypy() -> None:
    """Run mypy through pytest, raise error when problem."""
    # NOTE: Does not check the contents of stdout yet.
    # It may be possible to have an exitcode with only warnings / notes in the stdout?

    # Assume we want to use the pyproject.toml from two layers above this file
    tomldir = pathlib.Path(__file__).parent.parent
    tomlfile = tomldir / "pyproject.toml"
    assert tomlfile.exists()
    stdout, stderr, exitcode = api.run(["--config-file", str(tomlfile)])

    if stderr:
        raise RuntimeError("Error while running mypy through pytest:\n" + stderr)
    if exitcode:
        # Could make bit more informative by using the last line of the stdout
        # in the error summary
        raise AssertionError("The mypy run found problems in files\n" + stdout)
