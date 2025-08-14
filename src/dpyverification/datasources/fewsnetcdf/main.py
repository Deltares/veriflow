"""Read and write netcdf files in a fews compatible format."""

from enum import StrEnum
from typing import Self

import numpy as np
import xarray as xr

from dpyverification.configuration import FileInputFewsnetcdfConfig
from dpyverification.configuration.default.datasources import FewsNetcdfKind
from dpyverification.constants import SimObsKind, StandardCoord, StandardDim
from dpyverification.datasources.base import BaseDatasource


class FewsNetcdfDims(StrEnum):
    """List of dimension names."""

    analysis_time = "analysis_time"
    stations = "stations"


class FewsNetcdfCoord(StrEnum):
    """List of dimension names."""

    station_id = "station_id"
    station_names = "station_names"


class Preprocessor:
    """Used in xr.open_mfdataset(preprocess=preprocessor_instance)."""

    def __init__(
        self,
        fews_netcdf_kind: FewsNetcdfKind,
        filter_variables: list[str] | None = None,
        filter_stations: list[str] | None = None,
        filter_forecast_periods: list[np.timedelta64] | None = None,
    ) -> None:
        self.fews_netcdf_kind = fews_netcdf_kind
        self.variables = filter_variables
        self.stations = filter_stations
        self.forecast_periods = filter_forecast_periods

    @staticmethod
    def convert_byte_string_coord_to_utf8(
        dataset: xr.Dataset,
        coords: list[FewsNetcdfCoord],
    ) -> xr.Dataset:
        """Convert byte strings."""
        for coord in coords:
            dataset[coord] = xr.DataArray(
                [  # type:ignore[misc]
                    v.decode("utf-8") if isinstance(v, bytes) else v  # type:ignore[misc]
                    for v in dataset[coord].to_numpy()  # type:ignore[misc]
                ],
                dims=dataset[coord].dims,
            )
        return dataset

    @staticmethod
    def rename_dims_coords_to_internal(dataset: xr.Dataset, fews_netcdf_kind: str) -> xr.Dataset:
        """Rename dims, coords to internal definition."""
        # Rename station coords/dims
        dataset = dataset.rename({FewsNetcdfCoord.station_names: StandardCoord.station_name.name})  # type:ignore[misc]
        dataset = dataset.rename({FewsNetcdfDims.stations: StandardDim.station})  # type:ignore[misc]
        dataset = dataset.set_coords(StandardCoord.station_name.name)

        if fews_netcdf_kind == FewsNetcdfKind.one_full_simulation:
            # Rename analysis_time for simulations
            dataset = dataset.rename(
                {FewsNetcdfDims.analysis_time: StandardDim.forecast_reference_time},  # type:ignore[misc]
            )

        return dataset

    @staticmethod
    def filter_stations(dataset: xr.Dataset, stations: list[str]) -> xr.Dataset:
        """Filter stations."""
        swap_dict = {StandardDim.station: StandardCoord.station_id.name}
        dataset = dataset.swap_dims(swap_dict)
        dataset = dataset.sel({StandardCoord.station_id.name: stations})  # type:ignore[misc]
        return dataset.swap_dims({StandardCoord.station_id.name: StandardDim.station})  # type:ignore[misc]

    @staticmethod
    def transform_full_simulation_to_full_info_sim_dataset(
        dataset: xr.Dataset,
        filter_forecast_periods: list[np.timedelta64] | None,
    ) -> xr.Dataset:
        """Transform the dataset to full-information dataset."""
        forecast_periods = (dataset["time"] - dataset["forecast_reference_time"]).to_numpy().ravel()  # type:ignore[misc]
        ds = dataset.assign_coords({"forecast_period": ("time", forecast_periods)})  # type:ignore[misc]

        ds = ds.swap_dims({"time": "forecast_period"}).drop_vars("time")  # type:ignore[misc]

        # Re-compute time as 2d matrix along forecast_period and forecast_reference_time
        time_index_2d = (
            (ds["forecast_reference_time"] + ds["forecast_period"]).to_numpy().swapaxes(0, 1)  # type:ignore[misc]
        )

        # Now assign time
        ds = ds.assign_coords(
            {"time": (("forecast_period", "forecast_reference_time"), time_index_2d)},  # type:ignore[misc]
        )

        # Filter data variables explicitly
        for data_var in ds.data_vars:
            ds[data_var] = ds[data_var].expand_dims(
                {"forecast_reference_time": ds["forecast_reference_time"]},
            )

        if filter_forecast_periods is not None:
            # Filter the relevant forecast_periods to maximize memory efficiency
            selector = {"forecast_period": filter_forecast_periods}
            ds = ds.sel(selector)

        return ds

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """Sequence of processing tasks."""
        # Decode byte-string coords
        dataset = self.convert_byte_string_coord_to_utf8(
            dataset,
            coords=[FewsNetcdfCoord.station_id],
        )

        # Rename dims/coords to internal definitions
        dataset = self.rename_dims_coords_to_internal(
            dataset,
            fews_netcdf_kind=self.fews_netcdf_kind,
        )

        # Transform to full info sim dataset
        if self.fews_netcdf_kind == FewsNetcdfKind.one_full_simulation:
            dataset = self.transform_full_simulation_to_full_info_sim_dataset(
                dataset,
                filter_forecast_periods=self.forecast_periods,
            )

        # Filter variables
        if self.variables is not None:
            dataset = dataset[self.variables]

        # Filter stations
        if self.stations is not None:
            dataset = self.filter_stations(dataset, self.stations)

        return dataset


class FewsNetcdfFile(BaseDatasource):
    """For reading data from, and writing data to, a fews netcdf file."""

    kind = "fewsnetcdf"
    config_class = FileInputFewsnetcdfConfig

    def __init__(self, config: FileInputFewsnetcdfConfig) -> None:
        self.config: FileInputFewsnetcdfConfig = config

    @staticmethod
    def transform_to_forecast_period_dataset(dataset: xr.Dataset) -> xr.Dataset:
        """
        Transform dataset for pipeline ingestion.

        Transform so that 'forecast_period' is a dimension and 'time' remains
        a coordinate for explicit timestamps. This allows slicing by forecast_period
        while keeping the timestamp available.
        """
        for data_var in dataset.data_vars:
            da = dataset[data_var]

            da_list = []
            for forecast_period in da["forecast_period"]:  # type:ignore[misc]
                da_fp = da.sel({"forecast_period": forecast_period})  # type:ignore[misc]
                da_fp = da_fp.expand_dims({"forecast_period": 1})
                da_fp = da_fp.assign_coords(
                    {"forecast_period": ("forecast_period", forecast_period.to_numpy().reshape(1))},  # type:ignore[misc]
                )
                da_fp = da_fp.swap_dims({"forecast_reference_time": "time"})  # type:ignore[misc]
                da_list.append(da_fp)

        new_dataset = xr.combine_by_coords(da_list)

        if isinstance(new_dataset, xr.Dataset):  # type:ignore[misc]
            return new_dataset

        msg = (
            "Invalid resulting datatype after transforming to forecast period dataset.",
            "Expected xr.Dataset, got {type(new_dataset)}",
        )
        raise TypeError(msg)

    def get_data(self) -> Self:
        """Retrieve fewsnetcdf content as an xarray DataArray."""
        if self.config.simobskind == SimObsKind.combined:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        # Configure pre-processing
        preprocessor = Preprocessor(
            fews_netcdf_kind=self.config.netcdf_kind,
            filter_stations=self.config.station_ids,
            filter_forecast_periods=self.config.forecast_periods.timedelta64,
        )

        # Observations
        if self.config.simobskind == SimObsKind.obs:
            with xr.open_mfdataset(
                self.config.paths,  # type:ignore[arg-type] # generator is acceptable argument
                preprocess=preprocessor,
            ) as dataset:
                dataset.load()

            self.dataset = dataset
            return self

        # Simulations
        with xr.open_mfdataset(
            self.config.paths,  # type:ignore[arg-type] # generator is acceptable argument
            combine="nested",
            concat_dim="forecast_reference_time",
            preprocess=preprocessor,
            coords="minimal",
            compat="override",
        ) as dataset:
            dataset.load()

        # Load the dataset into memory
        #   for now, we assume the dataset with lead time filtering fits into memory
        dataset.load()

        # Make the dataset ready for pipeline ingestion
        self.dataset = FewsNetcdfFile.transform_to_forecast_period_dataset(dataset)
        return self
