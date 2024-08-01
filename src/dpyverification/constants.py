"""Collection of constant values definitions.

Both single values, and enum classes.
These values are can be used throughout DPyVerification.
"""

import pathlib
import subprocess
from enum import StrEnum

import importlib_metadata


class DataSourceTypeEnum(StrEnum):
    """Enumeration of the supported input and / or output datasource types."""

    pixml = "pixml"
    fewsnetcdf = "fewsnetcdf"
    fewswebservice = "fewswebservice"


class SimObsType(StrEnum):
    """Enumeration of the supported types of input data."""

    sim = "sim"
    obs = "obs"
    combined = "combined"


class CalculationTypeEnum(StrEnum):
    """Enumeration of the implemented verification calculations."""

    simobspairs = "simobspair"
    pinscore = "pinscore"


def _set_version_info() -> tuple[str, str]:
    version = importlib_metadata.version("dpyverification")
    version_extra = ""

    this_dir = pathlib.Path(__file__).parent
    command = ["git", "rev-parse", "HEAD"]
    completed = subprocess.run(command, cwd=this_dir, capture_output=True, check=False, text=True)  # noqa: S603 # No S603 since we are pretty certain the input is not dangerous
    if completed.returncode == 0:
        # Git is installed, and this file is in a directory that is part of a git repository
        # Add git commit hash as version_extra
        stdout = completed.stdout
        version_extra = "+" + stdout.strip()
        # Also check whether there are uncommitted changes on top of the commit
        command = ["git", "diff", "HEAD"]
        completed = subprocess.run(  # noqa: S603 # No S603 since we are pretty certain the input is not dangerous
            command,
            cwd=this_dir,
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.stdout:
            version_extra += ".dirty"
    else:
        origin = importlib_metadata.distribution("dpyverification").origin  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
        if (
            origin  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
            and hasattr(origin, "dir_info")  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
            and hasattr(origin.dir_info, "editable")  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
            and origin.dir_info.editable  # type: ignore[misc] # Since not sure what will be returned, there is the long if statement
        ):
            # Package installed in editable mode, but no way to check whether it has been edited
            version_extra = "+editable"

    version_full = version + version_extra
    return version, version_full


NAME = "DPyVerification"
VERSION, VERSION_FULL = _set_version_info()
