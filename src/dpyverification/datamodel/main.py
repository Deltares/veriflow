"""Module with the dpyverification internal DataModel."""

from collections.abc import Iterable
from enum import Enum
from typing import Literal

import xarray as xr

from dpyverification.configuration.utils import VerificationPair
from dpyverification.constants import StandardDim, TimeseriesKind
from dpyverification.datasources.inputschemas import input_schemas


class DatasetKind(Enum):
    """Supported types of input data."""

    SIMULATION = "SIMULATION"
    OBSERVATION = "OBSERVATION"


def validate_data_array(data_array: xr.DataArray) -> None:
    """Validate a datasource by validating the data to a Pydantic schema."""
    schema = input_schemas[data_array.attrs["timeseries_kind"]]  # type:ignore[misc]
    schema.model_validate(data_array.to_dict(data=False))  # type:ignore[misc]


class InputDataset:
    """
    Class containing simulations and observations.

    SimObsDataset has functionality to retrieve verification pairs for computation of scores
    per pair. It is the central object used in the verification pipeline.
    """

    def __init__(
        self,
        data: Iterable[xr.DataArray],
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
        for data_array in data:
            validate_data_array(data_array)

        # Merge input data with outer join as a first validation
        #   in matching
        dataset = xr.merge(
            data,
            compat="override",
        )

        # The xarray.merge leaves the attributes of an arbitrary xr.DataArray as global attrs. We
        # clear the attribute dict to keep the internal global attrs clean.
        dataset.attrs.clear()  # type:ignore[misc]

        # Because we merged multiple xr.DataArrays, from different sources into one xr.Dataset,
        #   transpose the xr.Dataset so that all dimensions of the data variables are aligned.
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
    ) -> TimeseriesKind:
        """Return the timeseries kinds for a verification pair."""
        return TimeseriesKind(
            self.dataset[verification_pair.sim].attrs["timeseries_kind"],  # type:ignore[misc]
        )


class OutputDataset:
    """The internal output dataset.

    Contains input data, results from verification scores and metadata.
    """

    def __init__(
        self,
        input_dataset: InputDataset,
    ) -> None:
        self.input_dataset = input_dataset
        self.scores: xr.Dataset = xr.Dataset()

    def add_score(self, score: xr.DataArray) -> None:
        """Add a score to the scores list."""
        if score.name in self.scores:
            msg = f"Cannot add score to OutputDataset. Score ({score}) is already present."
            raise ValueError(msg)
        self.scores[score.name] = score

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
        include_input_dataset: bool = True,
    ) -> xr.Dataset:
        """Get the output dataset."""
        scores_selection: xr.Dataset = (
            self.scores[scores] if isinstance(scores, list) else self.scores
        )

        return (
            xr.merge([scores_selection, self.input_dataset.dataset], combine_attrs="drop")  # type:ignore[misc]
            if include_input_dataset
            else scores_selection
        )
