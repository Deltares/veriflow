"""Specification of a pipeline that will collect data and run verification functions on the data."""

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

import xarray as xr

from dpyverification.configuration import Config, ConfigFile
from dpyverification.configuration.file import ConfigKind
from dpyverification.datamodel import InputDataset, OutputDataset
from dpyverification.datasinks.base import BaseDatasink
from dpyverification.datasinks.cf_compliant_netdf import CFCompliantNetCDF
from dpyverification.datasources.base import BaseDatasource
from dpyverification.datasources.fewsnetcdf import FewsNetCDF
from dpyverification.datasources.fewswebservice import FewsWebservice
from dpyverification.scores.base import BaseScore
from dpyverification.scores.probabilistic import CrpsForEnsemble, RankHistogram

TItem = TypeVar("TItem", bound=BaseDatasource | BaseDatasink | BaseScore)

DEFAULT_DATASOURCES: list[type[BaseDatasource]] = [
    FewsNetCDF,
    FewsWebservice,
]
DEFAULT_SCORES: list[type[BaseScore]] = [RankHistogram, CrpsForEnsemble]
DEFAULT_DATASINKS: list[type[BaseDatasink]] = [CFCompliantNetCDF]


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


def merge_user_and_default_items(
    default_items: list[type[TItem]],
    user_items: list[type[TItem]] | None,
) -> list[type[TItem]]:
    """Merge default and user-provided items."""
    if user_items is None:
        return default_items
    return default_items + user_items


def execute_pipeline(
    config: tuple[Path, ConfigKind] | Config,
    user_datasources: list[type[BaseDatasource]] | None = None,
    user_scores: list[type[BaseScore]] | None = None,
    user_datasinks: list[type[BaseDatasink]] | None = None,
) -> xr.Dataset:
    """Execute a verification pipeline as defined in the configuration.

    Parameters
    ----------
    config : tuple[Path, ConfigKind] | Config
        When using a configuration file, provide a tuple with the path and kind
        of configuration file. For now, only 'yaml' is supported.
    user_datasources : list[type[BaseDatasource]] | None, optional
        Option to plug-in a user-implementation of a DataSource., by default None
    user_scores : list[type[BaseScore]] | None, optional
        Option to plug-in a user-implementation of a Score., by default None
    user_datasinks : list[type[BaseDatasink]] | None, optional
        Option to plug-in a user-implementation of a DataSink., by default None

    Returns
    -------
    xr.Dataset
        The output dataset containing the results of the verification pipeline. In addition to the
        option of writing the output to a file or service, the output of the verification pipeline
        can also be assigned back to a Python variable for further inspection in an interactive
        Python environment.
    """
    # Get the available sources, scores and sinks
    available_datasources = merge_user_and_default_items(
        DEFAULT_DATASOURCES,
        user_datasources,
    )
    available_scores = merge_user_and_default_items(
        DEFAULT_SCORES,
        user_scores,
    )
    available_datasinks = merge_user_and_default_items(
        DEFAULT_DATASINKS,
        user_datasinks,
    )

    # Initialize the config instance from file when it's not directly provided
    if not isinstance(config, Config):
        config = ConfigFile(
            config_file=config[0],
            config_type=config[1],
        ).content

    # Collect and initialize all datasources
    datasources: list[BaseDatasource] = []
    for datasource_config in config.datasources:
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

    # Initialize the input dataset
    input_dataset = InputDataset(
        [datasource.data_array for datasource in datasources],
    )

    # Initialize the output dataset
    output_dataset = OutputDataset(input_dataset=input_dataset)

    # Add score results to the output dataset
    for score_config in config.scores:
        score_kind = find_matching_kind_in_list(
            items=available_scores,
            kind=score_config.kind,
        )
        score = score_kind.from_config(score_config.model_dump())  # type: ignore[misc] # Allow Any
        results = score.validate_and_compute(input_dataset)
        if isinstance(results, xr.DataArray):  # type: ignore[misc]
            output_dataset.add_score(results)
        elif isinstance(results, Iterable):
            for result in results:  # type: ignore[misc]
                output_dataset.add_score(result)  # type: ignore[misc]

    # Write data for each datasink if not None
    if config.datasinks is not None:
        for datasink_config in config.datasinks:
            sink_kind = find_matching_kind_in_list(
                items=available_datasinks,
                kind=datasink_config.kind,
            )
            datasink = sink_kind.from_config(datasink_config.model_dump())  # type: ignore[misc] # Allow Any
            datasink.write_data(
                output_dataset.get_output_dataset(),
            )

    # Return the output dataset by default
    return output_dataset.get_output_dataset()
