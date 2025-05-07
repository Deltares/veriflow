"""
Specification of a pipeline that will collect data and run verification functions on the data.

Specification is expected to be in a configuration file.
Results can at least be written to netcdf file.
"""

import pathlib
from typing import TypeVar

from dpyverification.configuration import ConfigFile
from dpyverification.datamodel import DataModel
from dpyverification.datasinks.base import BaseDatasink
from dpyverification.datasinks.fewsnetcdf import FewsNetcdfFileSink
from dpyverification.datasources.base import BaseDatasource
from dpyverification.datasources.fewsnetcdf.main import FewsNetcdfFile
from dpyverification.datasources.pixml import PiXmlFile
from dpyverification.scores.base import BaseScore
from dpyverification.scores.crps_for_ensemble import CrpsForEnsemble
from dpyverification.scores.rankhistogram import RankHistogram
from dpyverification.scores.simobspairs import SimObsPairs

TItem = TypeVar("TItem", bound=BaseDatasource | BaseDatasink | BaseScore)

DEFAULT_DATASOURCES: list[type[BaseDatasource]] = [PiXmlFile, FewsNetcdfFile]
DEFAULT_SCORES: list[type[BaseScore]] = [RankHistogram, CrpsForEnsemble, SimObsPairs]
DEFAULT_DATASINKS: list[type[BaseDatasink]] = [FewsNetcdfFileSink]


def find_matching_kind_in_list(
    items: list[type[TItem]],
    kind: str,
) -> type[TItem]:
    """Return a datasource, calcuation or datasink of a given kind."""
    for item in items:
        if kind == item.kind:
            return item
    msg = f"No item with type {kind} exists."
    raise ValueError(msg)


def execute_pipeline(
    configfile: pathlib.Path,
    configtype: str = "yaml",
    user_datasources: list[type[BaseDatasource]] | None = None,
    user_scores: list[type[BaseScore]] | None = None,
    user_datasinks: list[type[BaseDatasink]] | None = None,
) -> None:
    """Execute a pipeline as defined in the configfile."""
    # TODO(AU): Implement parsing of a runinfo xml file into a valid config dict # noqa: FIX002
    #   https://github.com/Deltares-research/DPyVerification/issues/8
    #   As part of that, add a unit test on what happens if a wrong conftype is passed, and make
    #   sure it gives a nice error message

    available_datasources = (
        user_datasources + DEFAULT_DATASOURCES
        if user_datasources is not None
        else DEFAULT_DATASOURCES
    )
    available_scores = user_scores + DEFAULT_SCORES if user_scores is not None else DEFAULT_SCORES
    available_datasinks = (
        user_datasinks + DEFAULT_DATASINKS if user_datasinks is not None else DEFAULT_DATASINKS
    )
    # Initialize the config instance
    config = ConfigFile(
        configfile,
        configtype,
    )

    # Collect and initialize all datasources
    datasources: list[BaseDatasource] = []
    for datasource_config in config.content.datasources:
        source_kind = find_matching_kind_in_list(
            items=available_datasources,
            kind=datasource_config.kind,
        )
        datasource = source_kind.from_config(
            datasource_config.model_dump(),  # type: ignore[misc] # Allow Any
        )
        datasources.append(datasource)

    # Get data for each datasource
    for datasource in datasources:
        datasource.get_data()

    # Initialize the datamodel
    datamodel = DataModel(datasources, config.content.general)

    # Add score results to the datamodel
    for score_config in config.content.scores:
        score_kind = find_matching_kind_in_list(
            items=available_scores,
            kind=score_config.kind,
        )
        score = score_kind.from_config(score_config.model_dump())  # type: ignore[misc] # Allow Any
        result = score.compute(datamodel)
        datamodel.add_to_output(result)

    # Write data for each datasink
    for datasink_config in config.content.datasinks:
        sink_kind = find_matching_kind_in_list(items=available_datasinks, kind=datasink_config.kind)
        datasink = sink_kind.from_config(datasink_config.model_dump())  # type: ignore[misc] # Allow Any
        datasink.write_data(datamodel.output)
