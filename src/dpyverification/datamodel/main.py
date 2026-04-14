"""Module with the dpyverification internal DataModel."""

from collections.abc import Iterable

import xarray as xr
from pydantic import ValidationError

from dpyverification.configuration.utils import VerificationPair
from dpyverification.constants import (
    FORECAST_DATA_TYPES,
    HISTORICAL_DATA_TYPES,
    DataType,
    StandardDim,
)
from dpyverification.datasources.inputschemas import INPUT_SCHEMAS

__all__ = ["InputDataset", "OutputDataset"]


@xr.register_dataarray_accessor("verification")  # type:ignore[no-untyped-call, misc]
class InputDataArrayExtension:
    """xr.DataArray representing specific data type.

    xr.register_dataset_accessor is the recommended way to extend xr.DataArray.
    see: https://docs.xarray.dev/en/stable/internals/extending-xarray.html. It's used there to
    extend the input data arrays so we can directly access properties (like data type and
    source) and the validation method that checks the input array against a schema.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    @property
    def data_type(self) -> str:
        """The data type of the array."""
        if "data_type" not in self._obj.attrs:  # type:ignore[misc]
            msg = f"No data type set on {self._obj} attrs."
            raise ValueError(msg)
        return DataType(self._obj.attrs["data_type"])  # type:ignore[misc]

    @property
    def is_thresholds(self) -> bool:
        """Boolean indicating this array is a thresholds array."""
        return self.data_type == DataType.threshold

    @property
    def is_historical(self) -> bool:
        """Boolean indicating this array is a historical."""
        return self.data_type in HISTORICAL_DATA_TYPES

    @property
    def is_forecast(self) -> bool:
        """Boolean indicating this array is a forecast."""
        return self.data_type in FORECAST_DATA_TYPES

    @property
    def source(self) -> str:
        """The source name."""
        return str(self._obj.name)

    def validate(self) -> None:
        """Validate the data according to schema."""
        schema = INPUT_SCHEMAS[self.data_type]  # type:ignore[index] # str is compatible with StrEnum index

        try:
            schema.model_validate(self._obj.to_dict(data=False))  # type:ignore[misc]
        except ValidationError as exc:
            msg = (f"Validation failed for data_type '{self.data_type}'.\n{exc}",)
            raise ValueError(msg) from exc


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
        """Initialize the InputDataset.

        Validates each input data array against a schema and collects all input data into a
        dictionary, keyed by the source and valued by the xr.DataArray.
        """
        self.datastore: dict[str, xr.DataArray] = {}

        # Validate, and add to datastore
        for data_array in data:
            data_array.verification.validate()  # type:ignore[misc]
            self.datastore[data_array.verification.source] = data_array  # type:ignore[misc]

    @staticmethod
    def map_historical_into_forecast_space(
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.DataArray:
        """
        Transform array of historical data into forecast structure.

        Given an observation array with dimension 'time' and a simulation array with
        dimensions 'forecast_reference_time' and 'forecast_period', project the observed
        values onto the simulation array.

        This method is called at runtime when the pipeline starts a score computation on forecast
        data. On the fly, the observation array is mapped to the forecast structure, so data are
        aligned along the same dimensions.
        """
        # Stack forecast time axes
        stacked_time = sim[StandardDim.time].stack(  # type:ignore[misc]
            z=(StandardDim.forecast_reference_time, StandardDim.forecast_period),
        )

        # Reindex observations onto stacked forecast times
        obs_aligned = obs.reindex(
            time=stacked_time.to_numpy(),  # type:ignore[misc]
        )

        # Attach forecast coordinates explicitly (from the MultiIndex)
        z_index = stacked_time.indexes["z"]  # type:ignore[misc]

        # Assign forecast_reference_time and forecast_period coordinates to the aligned
        # observations, based on the MultiIndex of the stacked time dimension. This is
        # necessary because after re-indexing, the original time dimension of the observations
        # is now aligned with the stacked time dimension of the simulations, which has a MultiIndex
        # of forecast_reference_time and forecast_period.
        obs_aligned = obs_aligned.assign_coords(
            forecast_reference_time=(  # type:ignore[misc]
                StandardDim.time,
                z_index.get_level_values(StandardDim.forecast_reference_time),  # type:ignore[misc]
            ),
            forecast_period=(  # type:ignore[misc]
                StandardDim.time,
                z_index.get_level_values(StandardDim.forecast_period),  # type:ignore[misc]
            ),
        )

        # Set the time coordinate to be the stacked time (MultiIndex of forecast_reference_time and
        # forecast_period)
        obs_indexed = obs_aligned.set_index(
            time=(StandardDim.forecast_reference_time, StandardDim.forecast_period),
        )

        # Unstack into forecast space
        obs_projected = obs_indexed.unstack(StandardDim.time)

        # Preserve attrs
        obs_projected.attrs = obs.attrs

        return obs_projected

    def get_pair(
        self,
        verification_pair: VerificationPair,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return observations and simulations for a given verification pair.

        This method is called by the verification pipeline at runtime to retrieve the correct data
        for one of the configured verification pairs.
        """
        obs = self.datastore[verification_pair.obs]
        sim = self.datastore[verification_pair.sim]

        if sim.verification.is_forecast:  # type:ignore[misc]
            # Map historical into forecast space upon score computation
            return self.map_historical_into_forecast_space(obs, sim), sim

        # If the simulation is not a forecast, it is a historical data type (an observation or
        #   historical simulation). In this case: verify along dimension 'time' instead of mapping
        #   data into forecast space.
        return obs, sim

    def get_thresholds_array(self) -> xr.DataArray:
        """Get the thresholds array from the input dataset."""
        for data_array in self.datastore.values():
            if data_array.verification.is_thresholds:  # type:ignore[misc]
                return data_array
        msg = (
            "No thresholds array found in the input dataset, but required for computing "
            "categorical scores."
        )
        raise ValueError(msg)


class OutputDataset:
    """The internal output dataset.

    Contains input data, results from verification scores and metadata.
    """

    def __init__(
        self,
        input_dataset: InputDataset,
    ) -> None:
        self.input_dataset = input_dataset

        # Internal datastore that stores results of score computation in a dictionary where the
        #   key represent the pair_id of the VerificationPair and the value is an xr.Dataset that
        #   contains all results from varying scores for that pair.
        self.datastore: dict[str, xr.Dataset] = {}

    def add_score(self, score: xr.DataArray | xr.Dataset, verification_pair_id: str) -> None:
        """Add a score results to the datastore."""
        # Convert to xr.Dataset
        if isinstance(score, xr.DataArray):  # type:ignore[misc]
            score = score.to_dataset()

        # Add to the store, if not added before
        if verification_pair_id not in self.datastore:
            self.datastore[verification_pair_id] = score

        # Pair has added data to the datastore before, so merge
        else:
            self.datastore[verification_pair_id] = xr.merge(
                [self.datastore[verification_pair_id], score],  # type:ignore[list-item, assignment]
            )

    def get_output_dataset(
        self,
        verification_pair: VerificationPair,
        *,
        include_input_data: bool = True,
    ) -> xr.Dataset:
        """Get the output dataset for a given verification pair."""
        if verification_pair.id in self.datastore:
            # Get the results for this pair
            dataset = self.datastore[verification_pair.id]

            if include_input_data:
                # Return results, include the input dataset
                obs, sim = self.input_dataset.get_pair(verification_pair)
                return xr.merge([obs, sim, dataset], compat="no_conflicts")  # type:ignore[list-item, return-value]

            # Return results, exclude input dataset
            return dataset

        # Return only input dataset (no results found in datastore)
        obs, sim = self.input_dataset.get_pair(verification_pair)
        return xr.merge([obs, sim], compat="no_conflicts")  # type:ignore[list-item, return-value]
