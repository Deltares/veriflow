"""
Specification of a pipeline that will collect data and run verification functions on the data.

Specification is expected to be in a configuration file.
Results can at least be written to netcdf file.
"""

import pathlib

from dpyverification.configuration import Config, ConfigTypes


def execute_pipeline(configfile: pathlib.Path, conf_type: str | None = "yaml") -> None:
    """Execute a pipeline as defined in the configfile."""
    conftype = ConfigTypes(
        conf_type,
    )
    # AU: does this give the possible values in the error, or at least the class?
    # Else use try-except for nice error message
    config = Config(configfile, conftype)

    # Until pipeline complete enough, as a last action mention the last-generated object
    _ = config
