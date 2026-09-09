"""Module with the veriflow internal DataModel."""

from collections.abc import Iterable

import xarray as xr

from veriflow.configuration.utils import VerificationPair
from veriflow.constants import (
    FORECAST_DATA_TYPES,
    HISTORICAL_DATA_TYPES,
    DataType,
    SpatialType,
    StandardAttribute,
    StandardDim,
)

__all__ = ["InputDataset"]


@xr.register_dataset_accessor("verification")  # type:ignore[no-untyped-call, misc]
class InputDatasetExtension:
    """xr.Dataset representing a specific data type from a specific source.

    xr.register_dataset_accessor is the recommended way to extend xr.Dataset.
    see: https://docs.xarray.dev/en/stable/internals/extending-xarray.html. It's used here to
    extend the input datasets so we can directly access properties (like data type and
    source) and the validation method that checks the input dataset against a schema.

    The dataset is expected to carry ``data_type`` and ``source`` keys on its ``attrs``.
    Each data variable in the dataset represents a physical variable, and is expected to
    carry a ``units`` key on its ``attrs``.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    @property
    def data_type(self) -> DataType:
        """The data type of the dataset."""
        if "data_type" not in self._obj.attrs:  # type:ignore[misc]
            msg = f"No data type set on {self._obj} attrs."
            raise ValueError(msg)
        return DataType(self._obj.attrs["data_type"])  # type:ignore[misc]

    @property
    def spatial_type(self) -> SpatialType:
        """The spatial type of the dataset (defaults to ``point`` when unset)."""
        return SpatialType(self._obj.attrs.get("spatial_type", SpatialType.point))  # type:ignore[misc]

    @property
    def is_thresholds(self) -> bool:
        """Boolean indicating this dataset is a thresholds dataset."""
        return self.data_type == DataType.threshold

    @property
    def is_historical(self) -> bool:
        """Boolean indicating this dataset is historical."""
        return self.data_type in HISTORICAL_DATA_TYPES

    @property
    def is_forecast(self) -> bool:
        """Boolean indicating this dataset is a forecast."""
        return self.data_type in FORECAST_DATA_TYPES

    @property
    def source_id(self) -> str:
        """The source ID."""
        if "source_id" not in self._obj.attrs:  # type:ignore[misc]
            msg = f"No source_id set on {self._obj} attrs."
            raise ValueError(msg)
        return str(self._obj.attrs["source_id"])  # type:ignore[misc]


class InputDataset:
    """
    Class containing simulations and observations.

    InputDataset has functionality to retrieve verification pairs for computation of scores
    per pair. It is the central object used in the verification pipeline.
    """

    def __init__(
        self,
        data: Iterable[xr.Dataset],
    ) -> None:
        """Initialize the InputDataset.

        Validates each input dataset against a schema and collects all input data into a
        dictionary, keyed by the source and valued by the xr.Dataset.
        """
        self.datastore: dict[str, xr.Dataset] = {}

        # Validate, and add to datastore
        for dataset in data:
            self.datastore[dataset.verification.source_id] = dataset  # type:ignore[misc]

    @staticmethod
    def map_historical_into_forecast_space(
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.DataArray:
        """
        Transform array of historical data into forecast structure.

        Given an observation array with dimension 'time' and a simulation array with
        dimensions 'forecast_reference_time' and 'lead_time', project the observed
        values onto the simulation array.

        This method is called at runtime when the pipeline starts a score computation on forecast
        data. On the fly, the observation array is mapped to the forecast structure, so data are
        aligned along the same dimensions.
        """
        # Stack forecast time axes
        stacked_time = sim[StandardDim.time].stack(
            z=(StandardDim.forecast_reference_time, StandardDim.lead_time),
        )

        # Reindex observations onto stacked forecast times
        obs_aligned = obs.reindex(
            time=stacked_time.to_numpy(),  # type:ignore[misc]
        )

        # Attach forecast coordinates explicitly (from the MultiIndex)
        z_index = stacked_time.indexes["z"]  # type:ignore[misc]

        # Assign forecast_reference_time and lead_time coordinates to the aligned
        # observations, based on the MultiIndex of the stacked time dimension. This is
        # necessary because after re-indexing, the original time dimension of the observations
        # is now aligned with the stacked time dimension of the simulations, which has a MultiIndex
        # of forecast_reference_time and lead_time.
        obs_aligned = obs_aligned.assign_coords(
            forecast_reference_time=(  # type:ignore[misc]
                StandardDim.time,
                z_index.get_level_values(StandardDim.forecast_reference_time),  # type:ignore[misc]
            ),
            lead_time=(  # type:ignore[misc]
                StandardDim.time,
                z_index.get_level_values(StandardDim.lead_time),  # type:ignore[misc]
            ),
        )

        # Set the time coordinate to be the stacked time (MultiIndex of forecast_reference_time and
        # lead_time)
        obs_indexed = obs_aligned.set_index(
            time=(StandardDim.forecast_reference_time, StandardDim.lead_time),
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

        Selects ``verification_pair.variable`` from each source's dataset and returns the
        resulting DataArrays. This method is called by the verification pipeline at runtime
        to retrieve the correct data for one of the configured verification pairs.
        """
        obs_ds = self.datastore[verification_pair.observations_source_id]
        sim_ds = self.datastore[verification_pair.simulations_source_id]

        variable = verification_pair.variable
        if variable not in obs_ds.data_vars:
            msg = (
                f"Variable '{variable}' configured on verification pair "
                f"'{verification_pair.id}' not found in observations source "
                f"'{verification_pair.observations_source_id}'. "
                f"Available variables: {sorted(obs_ds.data_vars)}."  # type:ignore[type-var]
            )
            raise ValueError(msg)
        if variable not in sim_ds.data_vars:
            msg = (
                f"Variable '{variable}' configured on verification pair "
                f"'{verification_pair.id}' not found in simulations source "
                f"'{verification_pair.simulations_source_id}'. Available variables: "
                f"{sorted(sim_ds.data_vars)}."  # type:ignore[type-var]
            )
            raise ValueError(msg)

        obs = obs_ds[variable]
        sim = sim_ds[variable]

        # Propagate dataset-level data_type onto each extracted DataArray, so downstream code
        # (scores etc.) can read it via the data array's attrs.
        obs.attrs.setdefault("data_type", obs_ds.attrs.get("data_type"))  # type:ignore[misc]
        sim.attrs.setdefault("data_type", sim_ds.attrs.get("data_type"))  # type:ignore[misc]
        # Likewise propagate spatial_type (defaulting to point) so the full data type is known.
        obs_spatial = obs_ds.attrs.get("spatial_type", SpatialType.point)  # type:ignore[misc]
        sim_spatial = sim_ds.attrs.get("spatial_type", SpatialType.point)  # type:ignore[misc]
        obs.attrs.setdefault("spatial_type", obs_spatial)  # type:ignore[misc]
        sim.attrs.setdefault("spatial_type", sim_spatial)  # type:ignore[misc]
        # Propagate the CRS so downstream reprojection knows the source CRS. Every validated
        # dataset is guaranteed to carry a ``crs`` attribute (defaulting to EPSG:4326).
        obs.attrs.setdefault(StandardAttribute.crs, obs_ds.attrs[StandardAttribute.crs])  # type:ignore[misc]
        sim.attrs.setdefault(StandardAttribute.crs, sim_ds.attrs[StandardAttribute.crs])  # type:ignore[misc]

        # Set the source id on the datasets
        obs.attrs.setdefault(  # type:ignore[misc]
            StandardAttribute.source_id,
            obs_ds.attrs.get(StandardAttribute.source_id),  # type:ignore[misc]
        )
        sim.attrs.setdefault(  # type:ignore[misc]
            StandardAttribute.source_id,
            sim_ds.attrs.get(StandardAttribute.source_id),  # type:ignore[misc]
        )

        if sim_ds.verification.is_forecast:  # type:ignore[misc]
            # Map historical into forecast space upon score computation
            return self.map_historical_into_forecast_space(obs, sim), sim

        # If the simulation is not a forecast, it is a historical data type (an observation or
        #   historical simulation). In this case: verify along dimension 'time' instead of mapping
        #   data into forecast space.
        return obs, sim

    def get_thresholds_array(self, variable: str) -> xr.DataArray:
        """Get the thresholds array for a given variable from the input dataset."""
        for dataset in self.datastore.values():
            if not dataset.verification.is_thresholds:  # type:ignore[misc]
                continue
            if variable not in dataset.data_vars:
                msg = (
                    f"Variable '{variable}' not found in thresholds dataset. "
                    f"Available variables: {sorted(dataset.data_vars)}."  # type:ignore[type-var]
                )
                raise ValueError(msg)
            return dataset[variable]
        msg = (
            "No thresholds dataset found in the input dataset, but required for computing "
            "categorical scores."
        )
        raise ValueError(msg)
