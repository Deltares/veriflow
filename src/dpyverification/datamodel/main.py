"""Module with the dpyverification internal DataModel."""

from collections.abc import Sequence
from enum import Enum
from typing import Literal

import xarray as xr
from pydantic import BaseModel, ValidationError

from dpyverification.configuration import GeneralInfoConfig
from dpyverification.configuration.utils import TimePeriod, VerificationPair
from dpyverification.constants import StandardCoord, StandardDim
from dpyverification.datasources.inputschemas import (
    Observation,
    Simulation,
)


class DatasetKind(Enum):
    """Supported types of input data."""

    SIMULATION = "SIMULATION"
    OBSERVATION = "OBSERVATION"


def transform_data_array(
    data_array: xr.DataArray,
    kind: DatasetKind,
    general_config: GeneralInfoConfig,
) -> xr.DataArray:
    """Transform a datasource to be compatible with the internal DataModel."""

    def clip_time_to_verification_period(
        data_array: xr.DataArray,
        verification_period: TimePeriod,
    ) -> xr.DataArray:
        """Clip the dataset on time dimension to verification period."""
        return data_array.sel(
            time=slice(verification_period.start, verification_period.end),
        )

    # For all datasets, set the station_id as index on station dim
    #   to ensure automatic alignment based on this coord later on.
    data_array = data_array.assign_coords(
        {
            StandardDim.station: data_array[StandardCoord.station.name].to_numpy(),  # type:ignore[misc]
        },
    )

    # Clip any input to be within the verification period
    data_array = clip_time_to_verification_period(
        data_array=data_array,
        verification_period=general_config.verification_period,
    )

    # Select only relevant forecast periods for simulations
    if kind == DatasetKind.SIMULATION:
        data_array = data_array.sel(
            forecast_period=general_config.forecast_periods.timedelta64,
        )

    return data_array


def validate_data_array(data_array: xr.DataArray) -> tuple[xr.DataArray, DatasetKind]:
    """Validate a datasource by validating the data to a Pydantic schema."""
    schemas: dict[type[BaseModel], DatasetKind] = {
        Observation: DatasetKind.OBSERVATION,
        Simulation: DatasetKind.SIMULATION,
    }

    def attempt_validation(
        schema: type[BaseModel],
        data_array: xr.DataArray,
    ) -> bool:
        """Validate and return True when successful, else False."""
        try:
            schema.model_validate(data_array.to_dict(data=False))  # type:ignore[misc]
            return True  # noqa: TRY300
        except ValidationError:
            return False

    for schema, kind in schemas.items():
        if attempt_validation(schema=schema, data_array=data_array):
            return data_array, kind
    msg = f"Invalid dataset {data_array}."
    raise ValidationError(msg)


class OutputDataset:
    """The internal output dataset.

    Contains input data, results from verification scores and metadata.
    """

    def __init__(
        self,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> None:
        self.obs: xr.DataArray = obs
        self.sim: xr.DataArray = sim
        self.scores: dict[str, xr.DataArray] = {}

    def add_score(self, kind: str, score: xr.DataArray) -> None:
        """Add a score to the scores list."""
        if kind in self.scores:
            msg = f"Cannot add score to OutputDataset. Score ({score}) is already present."
            raise ValueError(msg)
        self.scores[kind] = score

    def _get_score(self, kind: str) -> xr.DataArray:
        try:
            return self.scores[kind]
        except KeyError as e:
            msg = (
                f"Score kind ({kind}) not added to OutputDataset.",
                f"Available scores: ({self.scores.keys()})",
            )
            raise KeyError(msg) from e

    def get_output_dataset(
        self,
        scores: list[str] | Literal["all"] = "all",
        *,
        include_simobs: bool = True,
    ) -> xr.Dataset:
        """Get the output dataset."""
        scores_selection = (
            list(self.scores.values())
            if scores == "all"
            else [self._get_score(kind) for kind in scores]
        )

        if include_simobs:
            scores_selection.append(self.obs)
            scores_selection.append(self.sim)

        return xr.merge(scores_selection, combine_attrs="drop")


class SimObsDataset:
    """
    Class containing simulations and observations.

    SimObsDataset has functionality to retrieve verification pairs for computation of scores
    per pair. It is the central object used in the verification pipeline.
    """

    def __init__(
        self,
        data: Sequence[xr.DataArray],
        general_config: GeneralInfoConfig,
    ) -> None:
        """Initialize the SimObsDataset.

        Parameters
        ----------
        data : Sequence[xr.DataArray]
            A sequence of xr.DataArrays, representing either simulations
            or observations. The structure (dims, coords) of the array
            must be valid against pre-defined Pydantic schemas in
            :module:`dpyverification.datasources.inputschemas`
        general_config : GeneralInfoConfig
            General config for the pipeline. Used to clip data on the defined
            verification period.
        """
        # Validate input data
        validated_data_arrays = (validate_data_array(dataset) for dataset in data)

        # Transform datasets based on their type
        transformed_datasets = (
            transform_data_array(data_array, simulation_kind, general_config)
            for data_array, simulation_kind in validated_data_arrays
        )

        # Merge input data with outer join as a first validation
        #   in matching
        dataset = xr.merge(
            transformed_datasets,
            compat="override",
        )

        # Re-order shared dims in order
        dataset = dataset.transpose("source", "variable", "time", "station", ...)

        # Now assign obs and sim separately, to minimize memory on source dim
        self.obs = dataset["observations"]
        self.sim = dataset["simulations"]

    @staticmethod
    def inner_join_on_time(
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Align obs and sim along time dimension, keeping all other dims intact."""
        obs_aligned, sim_aligned = xr.align(obs, sim, join="outer")

        # Stack all dims except time to check for NaNs
        dims_to_check_obs = [d for d in obs_aligned.dims if d != StandardDim.time]
        dims_to_check_sim = [d for d in sim_aligned.dims if d != StandardDim.time]

        # True where there is at least one valid value along stacked dims
        obs_has_data = ~obs_aligned.stack(all_other=dims_to_check_obs).isnull().all("all_other")  # noqa: PD003, PD013
        sim_has_data = ~sim_aligned.stack(all_other=dims_to_check_sim).isnull().all("all_other")  # noqa: PD003, PD013

        # Determine masks
        valid_time = obs_has_data & sim_has_data

        # TODO(JB): Include statistics for missing values # noqa: FIX002
        # https://github.com/Deltares-research/DPyVerification/issues/82
        _ = obs_has_data & (~sim_has_data)
        _ = sim_has_data & (~obs_has_data)

        if len(valid_time) == 0:
            msg = (
                "Simulations and observations do not share any times.",
                f"For observed source: '{obs[StandardDim.source]}' and simulated source:",
                f"'{sim[StandardDim.source]}'.",
            )
            raise ValueError(msg)
        obs_valid = obs_aligned.sel({StandardDim.time: valid_time})  # type:ignore[misc]
        sim_valid = sim_aligned.sel({StandardDim.time: valid_time})  # type:ignore[misc]

        return obs_valid, sim_valid

    def get_verification_pair(
        self,
        verification_pair: VerificationPair,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return observations and simulations for a given verification pair."""
        obs = self.obs.sel(
            source=verification_pair.source.obs,
        )
        sim = self.sim.sel(
            source=verification_pair.source.sim,
        )

        # Only return time indexes for which both obs and sim are available not nan
        return self.inner_join_on_time(obs, sim)
