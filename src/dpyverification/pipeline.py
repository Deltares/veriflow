"""
Specification of a pipeline that will collect data and run verification functions on the data.

Specification is expected to be in a configuration file.
Results can at least be written to netcdf file.
"""

import pathlib

from dpyverification.configuration import Config, ConfigTypes
from dpyverification.datasources.pixml import PiXmlFile


def execute_pipeline(configfile: pathlib.Path, conf_type: str | None = "yaml") -> None:
    """Execute a pipeline as defined in the configfile."""
    conftype = ConfigTypes(
        conf_type,
    )
    # AU: does this give the possible values in the error, or at least the class?
    # Else use try-except for nice error message
    config = Config(configfile, conftype)

    datalist = []
    for datasource in config.datasources:
        # Might want to turn this if-elif into a mapping when many different datasourcetypes
        if datasource["datasourcetype"] == "pixml":
            datalist.append(PiXmlFile.get_data(datasource))
        else:
            raise NotImplementedError

    # Until pipeline complete enough, as a last action mention the last-generated object
    _ = datalist
