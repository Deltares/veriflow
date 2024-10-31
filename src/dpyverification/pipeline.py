"""
Specification of a pipeline that will collect data and run verification functions on the data.

Specification is expected to be in a configuration file.
Results can at least be written to netcdf file.
"""

import itertools
import pathlib

from dpyverification.configuration import (
    Config,
    ConfigTypes,
)
from dpyverification.constants import CalculationTypeEnum, DataSourceTypeEnum
from dpyverification.datamodel import DataModel
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile
from dpyverification.datasources.pixml import PiXmlFile
from dpyverification.verifications import simobspairs


def execute_pipeline(configfile: pathlib.Path, conf_type: str | None = "yaml") -> None:
    """Execute a pipeline as defined in the configfile."""
    conftype = ConfigTypes(
        conf_type,
    )
    # TODO(AU): Implement parsing of a runinfo xml file into a valid config dict # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/8
    #   As part of that, add a unit test on what happens if a wrong conftype is passed, and make
    #   sure it gives a nice error message
    config = Config(configfile, conftype)

    datalists = []
    for datasource in config.content.datasources:
        # Might want to turn this if-elif into a mapping when many different datasourcetypes
        if datasource.datasourcetype == DataSourceTypeEnum.pixml:
            datalists.append(PiXmlFile.get_data(datasource))
        else:
            # If an unknown datasource is used, error
            raise NotImplementedError
    datalist = list(itertools.chain.from_iterable(datalists))

    datamodel = DataModel(datalist)

    for calculation in config.content.calculations:
        if calculation.calculationtype == CalculationTypeEnum.simobspairs:
            datamodel.add_to_output(simobspairs.simobspairs(calculation, datamodel, config))
        else:
            # If an unknown calculation is used, error
            raise NotImplementedError

    for output in config.content.output:
        if output.datasourcetype == DataSourceTypeEnum.fewsnetcdf:
            FewsNetcdfFile.write_data(output, datamodel.output)
        else:
            # If an unknown output is specified, error
            raise NotImplementedError
