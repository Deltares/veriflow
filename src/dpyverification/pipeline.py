"""
Specification of a pipeline that will collect data and run verification functions on the data.

Specification is expected to be in a configuration file.
Results can at least be written to netcdf file.
"""

import itertools
import pathlib

from dpyverification.configuration import Config
from dpyverification.constants import CalculationType, DataSourceType
from dpyverification.datamodel import DataModel
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile
from dpyverification.datasources.pixml import PiXmlFile
from dpyverification.verifications import crps_for_ensemble, rankhistogram, simobspairs


def execute_pipeline(configfile: pathlib.Path, configtype: str = "yaml") -> None:
    """Execute a pipeline as defined in the configfile."""
    # TODO(AU): Implement parsing of a runinfo xml file into a valid config dict # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/8
    #   As part of that, add a unit test on what happens if a wrong conftype is passed, and make
    #   sure it gives a nice error message
    config = Config(configfile, configtype)

    datalists = []
    for datasource in config.content.datasources:
        # Might want to turn this if-elif into a mapping when many different datasourcetypes
        if datasource.datasourcetype == DataSourceType.PIXML:
            datalists.append(PiXmlFile.get_data(datasource))
        else:
            # If an unknown datasource is used, error
            raise NotImplementedError
    datalist = list(itertools.chain.from_iterable(datalists))

    datamodel = DataModel(datalist, config.content.general)

    for calculation in config.content.calculations:
        if calculation.calculationtype == CalculationType.SIMOBSPAIRS:
            datamodel.add_to_output(
                simobspairs.simobspairs(calculation, datamodel),
            )
        elif calculation.calculationtype == CalculationType.RANKHISTOGRAM:
            """datamodel.add_to_output(
                rankhistogram.rankhistogram(calculation, datamodel),
            )"""
            _ = rankhistogram
            msg = "Writing rankhistogram to output is not yet supported."
            raise NotImplementedError(msg)
        elif calculation.calculationtype == CalculationType.CRPSForEnsemble:
            datamodel.add_to_output(
                crps_for_ensemble.crps_for_ensemble(calculation, datamodel),
            )
        else:
            # If an unknown calculation is used, error
            raise NotImplementedError

    for output in config.content.output:
        if output.datasourcetype == DataSourceType.FEWSNETCDF:
            FewsNetcdfFile.write_data(output, datamodel.output)
        else:
            # If an unknown output is specified, error
            raise NotImplementedError
