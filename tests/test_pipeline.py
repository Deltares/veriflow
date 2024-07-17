"""Test the functions in the pipeline module."""

from pathlib import Path

import pytest
from dpyverification import pipeline

from tests import TESTS_CONFIGURATION_FILE


@pytest.mark.parametrize(("cfile", "ctype"), [("", "runinfo"), (TESTS_CONFIGURATION_FILE, "yaml")])
def test_execute_pipeline_happy(cfile: str, ctype: str) -> None:
    """Test at least one valid conf file for each conf type."""
    pipeline.execute_pipeline(Path(cfile), conf_type=ctype)


def test_execute_pipeline_bad_conf_type() -> None:
    """Check that error on invalid conf type."""
    with pytest.raises(ValueError, match="'1234567890' is not a valid ConfigTypes"):
        pipeline.execute_pipeline(Path(), conf_type="1234567890")
