"""Module for testing veriflow.constants."""

import subprocess

import pytest

from veriflow import constants


def test_set_version_info_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _set_version_info when git is not available."""

    def raise_file_not_found(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", raise_file_not_found)

    version, version_full = constants._set_version_info()

    assert version_full.startswith(version)
