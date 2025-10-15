"""Module with the dpyverification internal DataModel."""

from collections.abc import Sequence
from enum import Enum
from typing import Literal

import xarray as xr

from dpyverification.configuration.utils import VerificationPair
from dpyverification.constants import StandardDim
from dpyverification.datasources.inputschemas import input_schemas


class DatasetKind(Enum):
    """Supported types of input data."""

    SIMULATION = "SIMULATION"
    OBSERVATION = "OBSERVATION"


def validate_data_array(data_array: xr.DataArray) -> xr.DataArray:
    """Validate a datasource by validating the data to a Pydantic schema."""
    schema = input_schemas[data_array.attrs["timeseries_kind"]]  # type:ignore[misc]
    schema.model_validate(data_array.to_dict(data=False))  # type:ignore[misc]
    return data_array


class OutputDataset:
    """The internal output dataset.

    Contains input data, results from verification scores and metadata.
    """

    def __init__(
        self,
    ) -> None:
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
        input_dataset: xr.Dataset | None = None,
    ) -> xr.Dataset:
        """Get the output dataset."""
        scores_selection: list[xr.DataArray] = (
            list(self.scores.values())
            if scores == "all"
            else [self._get_score(kind) for kind in scores]
        )

        if input_dataset:
            scores_selection.append(input_dataset)  # type:ignore[arg-type] # it's ok to add a dataset to this list, merge will convert all inputs to datasets

        return xr.merge(scores_selection, combine_attrs="drop")


class InputDataset:
    """
    Class containing simulations and observations.

    SimObsDataset has functionality to retrieve verification pairs for computation of scores
    per pair. It is the central object used in the verification pipeline.
    """

    def __init__(
        self,
        data: Sequence[xr.DataArray],
    ) -> None:
        """Initialize the SimObsDataset.

        Parameters
        ----------
        data : Sequence[xr.DataArray]
            A sequence of xr.DataArrays, representing either simulations
            or observations. The structure (dims, coords) of the array
            must be valid against pre-defined Pydantic schemas in
            :module:`dpyverification.datasources.inputschemas`
        """
        # Validate input data
        validated_data_arrays = (validate_data_array(data_array) for data_array in data)

        # Merge input data with outer join as a first validation
        #   in matching
        dataset = xr.merge(
            validated_data_arrays,
            compat="override",
        )

        # Empty attrs on the dataset level, while keeping attrs on individual variables
        #   in the dataset, that have the required 'timeseries_kind' attr.
        dataset.attrs.clear()  # type:ignore[misc]

        # Re-order shared dims in order
        self.dataset = dataset.transpose("variable", "time", "station", ...)

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
        obs = self.dataset[verification_pair.obs]
        sim = self.dataset[verification_pair.sim]

        # Only return time indexes for which both obs and sim are available not nan
        return self.inner_join_on_time(obs, sim)

    def get_simulated_timeseries_kind_from_pair(
        self,
        verification_pair: VerificationPair,
    ) -> str:
        """Return the timeseries kinds for a verification pair."""
        return str(  # type:ignore[misc]
            self.dataset[verification_pair.sim].attrs["timeseries_kind"],  # type:ignore[misc]
        )
