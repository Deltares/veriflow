"""Specification of a pipeline that will collect data and run verification functions on the data."""

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar, cast

import xarray as xr
from cftime import CFWarning  # type:ignore[import-untyped]
from xarray import SerializationWarning

from veriflow.cache import ZarrCache
from veriflow.configuration.config import Config
from veriflow.configuration.file import ConfigFile, ConfigKind
from veriflow.constants import StandardAttribute
from veriflow.datasinks import DEFAULT_DATASINKS
from veriflow.datasinks.base import BaseDatasink
from veriflow.datasources import DEFAULT_DATASOURCES
from veriflow.datasources.base import BaseDatasource
from veriflow.datatree.datatree import VeriflowDataTree
from veriflow.scores import DEFAULT_SCORES
from veriflow.scores.base import BaseCategoricalScore, BaseScore
from veriflow.transformations import parse_crs, project_to_crs

__all__ = ["run_pipeline"]

logger = logging.getLogger(__name__)


TItem = TypeVar("TItem", bound=BaseDatasource | BaseDatasink | BaseScore | BaseCategoricalScore)


def _align_crs(
    obs: "xr.DataArray",
    sim: "xr.DataArray",
    target_crs: str | None,
) -> "tuple[xr.DataArray, xr.DataArray]":
    """Align the CRS of ``obs`` and ``sim`` for score computation.

    When ``target_crs`` is set, both are reprojected to it (deriving ``x``/``y`` in the target
    CRS as needed) so the score is computed and expressed in that CRS. When ``target_crs`` is
    ``None``, ``obs`` and ``sim`` must share the same CRS, otherwise a ``ValueError`` is raised.

    Every input dataset is guaranteed to carry a ``crs`` attribute (defaulting to EPSG:4326
    during schema validation), so it is always present on the extracted DataArrays here.
    """
    if target_crs is not None:
        return project_to_crs(obs, target_crs), project_to_crs(sim, target_crs)

    obs_crs = cast("str", obs.attrs[StandardAttribute.crs])  # type: ignore[misc]
    sim_crs = cast("str", sim.attrs[StandardAttribute.crs])  # type: ignore[misc]
    # Fast path: identical CRS strings need no reprojection and no pyproj. Only when the
    # strings differ do we parse them (requiring pyproj) to check for semantic equality.
    if obs_crs != sim_crs and parse_crs(obs_crs) != parse_crs(sim_crs):  # type:ignore[misc]
        msg = (
            f"Observation CRS ('{obs_crs}') and simulation CRS ('{sim_crs}') differ, but no "
            "target CRS is configured on the score. Set 'crs' on the score configuration to "
            "reproject both to a common CRS."
        )
        raise ValueError(msg)
    return obs, sim


def find_matching_kind_in_list(
    items: Sequence[type[TItem]],
    kind: str,
) -> type[TItem]:
    """Return a datasource, calculation or datasink of a given kind."""
    for item in items:
        if kind == item.kind:
            return item
    msg = f"No item with type {kind} exists."
    raise ValueError(msg)


def merge_user_and_default_items(
    default_items: Sequence[type[TItem]],
    user_items: Sequence[type[TItem]] | None,
) -> list[type[TItem]]:
    """Merge default and user-provided items."""
    if user_items is None:
        return list(default_items)
    return list(default_items) + list(user_items)


def run_pipeline(
    config: tuple[Path, ConfigKind] | Config,
    user_datasources: list[type[BaseDatasource]] | None = None,
    user_scores: list[type[BaseScore] | type[BaseCategoricalScore]] | None = None,
    user_datasinks: list[type[BaseDatasink]] | None = None,
) -> VeriflowDataTree:
    """Execute a verification pipeline as defined in the configuration.

    Parameters
    ----------
    config : tuple[Path, ConfigKind] | Config
        When using a configuration file, provide a tuple with the path and kind
        of configuration file. For now, only 'yaml' is supported.
    user_datasources : list[type[BaseDatasource]] | None, optional
        Option to plug-in a user-implementation of a DataSource., by default None
    user_scores : list[type[BaseScore | BaseCategoricalScore]] | None, optional
        Option to plug-in a user-implementation of a Score., by default None
    user_datasinks : list[type[BaseDatasink]] | None, optional
        Option to plug-in a user-implementation of a DataSink., by default None

    Returns
    -------
    VeriflowDataTree
        The output datatree containing the results of the verification pipeline. In addition to the
        option of writing the output to a file or service, the output of the verification pipeline
        can also be assigned back to a Python variable for further inspection in an interactive
        Python environment.

    Examples
    --------
    Using a YAML file:

    .. code-block:: python

        from veriflow import run_pipeline
        from veriflow.configuration import Config
        from pathlib import Path

        path_to_config = Path("./config.yaml")
        output_datatree = run_pipeline((path_to_config, "yaml"))


    Using Python objects directly:

    .. code-block:: python

        from veriflow import run_pipeline
        from veriflow.configuration import Config, GeneralInfoConfig

        config = Config(
            general=GeneralInfoConfig(log_level="INFO"),
            # ... other sub-models here ...
        )

        output_datatree = run_pipeline(config)

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

    # Initialize the cache if caching is enabled in the configuration
    cache = ZarrCache(config.general.cache) if config.general.cache is not None else None

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

        # If caching is enabled, assign the cache to the datasource, so it can use it to store and
        # retrieve data
        datasource.cache = cache

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
            datasource.get_data()
            msg = (
                f"Dataset (source_id={datasource.config.source_id}) successfully loaded and "
                "validated."
            )
            logger.info(msg)

        # Initialize the output datatree and load the raw input data into it
        dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow-datatree"))
        dt.veriflow.add_input_data(
            [datasource.dataset for datasource in datasources],
        )

        msg = "Successfully loaded all data from sources."
        logger.info(msg)

        for verification_pair in config.general.verification_pairs:
            obs, sim = dt.veriflow.get_pair(verification_pair)
            dt.veriflow.add_staged_input_data(
                verification_pair=verification_pair,
                obs=obs,
                sim=sim,
            )

        # Add score results to the output dataset
        for score_config in config.scores:
            score_kind = find_matching_kind_in_list(
                items=available_scores,
                kind=score_config.score_adapter,
            )
            score = cast(
                "BaseScore | BaseCategoricalScore",
                score_kind.from_config(
                    score_config.model_dump(),  # type: ignore[misc] # Allow Any
                ),
            )
            for verification_pair in score.config.verification_pairs:
                obs, sim = dt.veriflow.get_pair(verification_pair)

                # Align the CRS of obs and sim. When a target CRS is configured on the score,
                # reproject both to it (results are then expressed in that CRS). Otherwise, obs
                # and sim must already share the same CRS.
                obs, sim = _align_crs(obs, sim, score.config.crs)

                # Check if the score is a categorical score, because in that case we need to provide
                # the thresholds array as well. We do this runtime check, because the contract of
                # the compute function in the BaseCategoricalScore is different from the one in
                # BaseScore, and we want to keep the compute function signature of BaseScore simple
                # without optional arguments that are only required for categorical scores.
                if isinstance(score, BaseCategoricalScore):
                    thresholds = dt.veriflow.get_thresholds_array(
                        verification_pair.variable,
                    )
                    result = score.validate_and_compute(obs=obs, sim=sim, thresholds=thresholds)
                else:
                    result = score.validate_and_compute(obs=obs, sim=sim)

                # Add the output of the score to the output dataset
                dt.veriflow.add_score(
                    verification_pair=verification_pair,
                    result=result,
                    name=score_config.score_adapter,
                )

                msg = (
                    f"Successfully computed {score.__class__.__name__} for verification pair "
                    f"{verification_pair.id}."
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
                datasink.write_data(dt)
                msg = f"Successfully wrote data using datasink {datasink_config.export_adapter}."
                logger.info(msg)

    msg = "Verification pipeline completed successfully."
    logger.info(msg)

    # Return the output dataset by default
    return dt
