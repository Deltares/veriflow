"""Specification of a pipeline that will collect data and run verification functions on the data."""

import logging
import warnings
from pathlib import Path
from typing import TypeVar

from cftime import CFWarning  # type:ignore[import-untyped]
from xarray import SerializationWarning

from dpyverification.configuration import Config, ConfigFile
from dpyverification.configuration.file import ConfigKind
from dpyverification.datamodel import InputDataset, OutputDataset
from dpyverification.datasinks.base import BaseDatasink
from dpyverification.datasinks.cf_compliant_netcdf import CFCompliantNetCDF
from dpyverification.datasources.base import BaseDatasource
from dpyverification.datasources.fewsnetcdf import FewsNetCDF
from dpyverification.datasources.fewswebservice import FewsWebservice
from dpyverification.scores.base import BaseScore
from dpyverification.scores.categorical import CategoricalScores
from dpyverification.scores.continuous import ContinuousScores
from dpyverification.scores.probabilistic import CrpsForEnsemble, RankHistogram

__all__ = ["execute_pipeline"]

logger = logging.getLogger(__name__)


TItem = TypeVar("TItem", bound=BaseDatasource | BaseDatasink | BaseScore)

DEFAULT_DATASOURCES: list[type[BaseDatasource]] = [
    FewsNetCDF,
    FewsWebservice,
]
DEFAULT_SCORES: list[type[BaseScore]] = [
    RankHistogram,
    CrpsForEnsemble,
    ContinuousScores,
    CategoricalScores,
]
DEFAULT_DATASINKS: list[type[BaseDatasink]] = [CFCompliantNetCDF]


def find_matching_kind_in_list(
    items: list[type[TItem]],
    kind: str,
) -> type[TItem]:
    """Return a datasource, calculation or datasink of a given kind."""
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
) -> OutputDataset:
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
    OutputDataset
        The output dataset containing the results of the verification pipeline. In addition to the
        option of writing the output to a file or service, the output of the verification pipeline
        can also be assigned back to a Python variable for further inspection in an interactive
        Python environment.

    Examples
    --------
    Using a YAML file:

    .. code-block:: python

        from dpyverification import execute_pipeline
        from dpyverification.configuration import Config
        from pathlib import Path

        path_to_config = Path("./config.yaml)
        output_dataset = execute_pipeline((path_to_config, "yaml"))


    Using Python objects directly:

    .. code-block:: python

        from dpyverification import execute_pipeline
        from dpyverification.configuration import Config, GeneralInfoConfig

        config = Config(
            general=GeneralInfoConfig(log_level="INFO"),
            # ... other sub-models here ...
        )

        output_dataset = execute_pipeline(config)

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

    # Log start message
    msg = (
        "Successfully initialized the configuration. \n\t verification_period_start = "
        f"{config.general.verification_period.start} \n\t verification_period_end = "
        f"{config.general.verification_period.end}"
    )
    logger.info(msg)

    # Collect and initialize all datasources
    datasources: list[BaseDatasource] = []
    for datasource_config in config.datasources:
        source_kind = find_matching_kind_in_list(
            items=available_datasources,
            kind=datasource_config.import_adapter,
        )
        datasource = source_kind.from_config(
            datasource_config.model_dump(),  # type: ignore[misc] # Allow Any
        )
        datasources.append(datasource)

    with warnings.catch_warnings():
        # Filter some known and harmless warnings
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in cast",
        )
        warnings.filterwarnings(
            "ignore",
            category=CFWarning,  # type:ignore[misc]
            message="this date/calendar/year zero convention is not supported by CF",
        )
        warnings.filterwarnings(
            "ignore",
            category=SerializationWarning,
            message="Unable to decode time axis into full numpy.datetime64 objects",
        )

        # Get data for each datasource
        for datasource in datasources:
            msg = f"Start getting data from {datasource.__class__.__name__}."
            logger.info(msg)
            datasource.get_data()
            msg = f"Successfully got data from {datasource.__class__.__name__}."
            logger.info(msg)

        # Initialize the input dataset
        input_dataset = InputDataset(
            [datasource.data_array for datasource in datasources],
        )

        msg = "Successfully loaded all data from sources."
        logger.info(msg)

        # Initialize the output dataset
        output_dataset = OutputDataset(input_dataset=input_dataset)

        # Add score results to the output dataset
        for score_config in config.scores:
            score_kind = find_matching_kind_in_list(
                items=available_scores,
                kind=score_config.score_adapter,
            )
            score = score_kind.from_config(score_config.model_dump())  # type: ignore[misc] # Allow Any
            for verification_pair in score.config.verification_pairs:
                obs, sim = input_dataset.get_pair(verification_pair)
                result = score.validate_and_compute(obs=obs, sim=sim)
                output_dataset.add_score(verification_pair_id=verification_pair.id, score=result)

                msg = (
                    f"Successfully computed {score.__class__.__name__} for verification pair "
                    "{pair_id}."
                )
                logger.info(msg)

        # Write data for each datasink if not None
        if config.datasinks is not None:
            for datasink_config in config.datasinks:
                sink_kind = find_matching_kind_in_list(
                    items=available_datasinks,
                    kind=datasink_config.export_adapter,
                )
                datasink = sink_kind.from_config(datasink_config.model_dump())  # type: ignore[misc] # Allow Any

                # We write results for each verification pair separately to the datasink. The
                #   datasink determines what the output will ook like.
                for verification_pair in config.general.verification_pairs:
                    datasink.write_data(
                        output_dataset.get_output_dataset(verification_pair),
                    )
                    msg = (
                        f"Successfully wrote results of verification pair {verification_pair.id} "
                        f"to {datasink.__class__.__name__}."
                    )
                    logger.info(msg)

    msg = "Verification pipeline completed successfully."
    logger.info(msg)

    # Return the output dataset by default
    return output_dataset
