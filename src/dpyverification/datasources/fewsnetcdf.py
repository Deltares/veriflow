"""Read and write NetCDF files in a fews compatible format."""

from enum import StrEnum
from typing import ClassVar, Self

import numpy as np
import xarray as xr

from dpyverification.configuration import FewsNetCDFConfig
from dpyverification.configuration.default.datasources import FewsNetCDFKind
from dpyverification.constants import StandardCoord, StandardDim, TimeseriesKind
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
        fews_netcdf_kind: FewsNetCDFKind,
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
    def rename_dims_coords_to_internal(
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Rename dims, coords to internal definition."""
        # Rename station coords/dims
        dataset = dataset.rename(
            {
                FewsNetcdfCoord.station_names: StandardCoord.station_name.name,
                FewsNetcdfCoord.station_id: StandardCoord.station.name,
            },  # type:ignore[misc]
        )
        dataset = dataset.swap_dims(
            {FewsNetcdfDims.stations: StandardDim.station},  # type:ignore[misc]
        )

        # Only the case when retrieving full forecasts (per forecast reference time)
        if StandardCoord.station_name.name in dataset:
            dataset = dataset.set_coords(StandardCoord.station_name.name)

        return dataset

    @staticmethod
    def filter_stations(
        dataset: xr.Dataset,
        stations: list[str],
    ) -> xr.Dataset:
        """Filter stations."""
        swap_dict = {StandardDim.station: StandardCoord.station.name}
        dataset = dataset.swap_dims(swap_dict)
        dataset = dataset.sel({StandardCoord.station.name: stations})  # type:ignore[misc]
        return dataset.swap_dims(
            {StandardCoord.station.name: StandardDim.station},  # type:ignore[misc]
        )

    @staticmethod
    def transform_full_simulation_to_full_info_sim_dataset(
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Transform the dataset to full-information dataset."""
        forecast_periods = (
            (dataset[StandardDim.time] - dataset[StandardDim.forecast_reference_time])
            # type:ignore[misc]
            .to_numpy()
            .ravel()
        )
        ds = dataset.assign_coords(
            {StandardDim.forecast_period: (StandardDim.time, forecast_periods)},  # type:ignore[misc]
        )

        ds = ds.swap_dims({StandardDim.time: StandardDim.forecast_period}).drop_vars(  # type:ignore[misc]
            StandardDim.time,
        )

        # Re-compute time as 2d matrix along forecast_period and forecast_reference_time
        time_index_2d = (
            (ds[StandardDim.forecast_reference_time] + ds[StandardDim.forecast_period])
            # type:ignore[misc]
            .to_numpy()
            .swapaxes(0, 1)
        )

        # Now assign time
        ds = ds.assign_coords(
            {
                StandardDim.time: (  # type:ignore[misc]
                    (StandardDim.forecast_period, StandardDim.forecast_reference_time),
                    time_index_2d,  # type:ignore[misc]
                ),
            },
        )

        # Set forecast reference time as dim on all vars
        for data_var in ds.data_vars:
            ds[data_var] = ds[data_var].expand_dims(
                {StandardDim.forecast_reference_time: ds[StandardDim.forecast_reference_time]},
            )

        return ds

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """Sequence of processing tasks."""
        # Decode byte-string coords
        dataset = Preprocessor.convert_byte_string_coord_to_utf8(
            dataset,
            coords=[FewsNetcdfCoord.station_id],
        )

        # Rename dims/coords to internal definitions
        dataset = Preprocessor.rename_dims_coords_to_internal(
            dataset,
        )

        if self.fews_netcdf_kind == FewsNetCDFKind.simulated_forecast_per_forecast_reference_time:
            # Rename analysis_time for simulations
            dataset = dataset.rename(
                {
                    FewsNetcdfDims.analysis_time: StandardDim.forecast_reference_time,  # type:ignore[misc]
                },
            )
            # Transform to full info sim dataset
            dataset = self.transform_full_simulation_to_full_info_sim_dataset(
                dataset,
            )

        # Filter variables
        if self.variables is not None:
            dataset = dataset[self.variables]

        # Filter stations
        if self.stations is not None:
            dataset = self.filter_stations(dataset, self.stations)

        # Filter forecast periods for simulations
        if (
            self.forecast_periods is not None
            and self.fews_netcdf_kind
            == FewsNetCDFKind.simulated_forecast_per_forecast_reference_time
        ):
            # Filter the relevant forecast_periods to maximize memory efficiency
            selector = {StandardDim.forecast_period: self.forecast_periods}
            dataset = dataset.sel(selector)

        return dataset


class FewsNetCDF(BaseDatasource):
    """For reading data from, and writing data to, a FEWS NetCDF file."""

    kind = "fewsnetcdf"
    config_class = FewsNetCDFConfig
    supported_timeseries_kinds: ClassVar[set[TimeseriesKind]] = {
        TimeseriesKind.observed_historical,
        TimeseriesKind.simulated_forecast_ensemble,
    }

    def __init__(self, config: FewsNetCDFConfig) -> None:
        self.config: FewsNetCDFConfig = config

    @staticmethod
    def transform_frt_simulation_to_internal_datamodel(
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """
        Transform dataset for pipeline ingestion.

        Transform so that 'forecast_period' is a dimension and 'time' remains
        a coordinate for explicit timestamps. This allows slicing by forecast_period
        while keeping the timestamp available.
        """
        data_arrays = []
        for data_var in dataset.data_vars:
            da = dataset[data_var]

            forecast_period_arrays = []
            for forecast_period in da[StandardDim.forecast_period]:  # type:ignore[misc]
                da_fp = da.sel({StandardDim.forecast_period: forecast_period})  # type:ignore[misc]
                da_fp = da_fp.expand_dims({StandardDim.forecast_period: 1})
                da_fp = da_fp.assign_coords(
                    {
                        StandardDim.forecast_period: (  # type:ignore[misc]
                            StandardDim.forecast_period,
                            forecast_period.to_numpy().reshape(1),  # type:ignore[misc]
                        ),
                    },
                )
                da_fp = da_fp.swap_dims({StandardDim.forecast_reference_time: StandardDim.time})  # type:ignore[misc]
                forecast_period_arrays.append(da_fp)

            data_arrays.append(
                xr.combine_nested(
                    forecast_period_arrays,
                    concat_dim=StandardDim.forecast_period,
                    combine_attrs="drop_conflicts",
                ),
            )

        dataset = xr.merge(data_arrays)

        # Reset and re-compute the forecast_reference_time to have no missing values on coord
        return dataset.assign_coords(
            {  # type:ignore[misc]
                StandardDim.forecast_reference_time: (  # type:ignore[misc]
                    (StandardDim.time, StandardDim.forecast_period),
                    (dataset[StandardDim.time] - dataset[StandardDim.forecast_period]).to_numpy(),  # type:ignore[misc]
                ),
            },
        )

    @staticmethod
    def convert_to_data_array_and_set_variable_and_units_coords(
        dataset: xr.Dataset,
        source: str,
        timeseries_kind: TimeseriesKind,
    ) -> xr.DataArray:
        """Transform dataset to internal datamodel."""
        # Extract the variable units from data variables
        units = [dataset[da].attrs["units"] for da in dataset]  # type:ignore[misc]

        # Stack the variables along dimension variable
        da = dataset.to_dataarray(dim=StandardDim.variable, name=source)

        # Set the configured timeseries kind as attribute
        da.attrs["timeseries_kind"] = timeseries_kind  # type:ignore[misc]

        # Set the station_id as index on station dim
        #   to ensure automatic alignment based on this coord later on.
        da = da.assign_coords(
            {
                StandardDim.station: da[StandardCoord.station.name].to_numpy(),  # type:ignore[misc]
            },
        )
        # Set the units as auxillary coordinate on new dimension variable
        return da.assign_coords(
            {StandardCoord.units.name: (StandardDim.variable, units)},  # type:ignore[misc]
        )

    def fetch_data(self) -> Self:
        """Retrieve fewsnetcdf content as an xarray DataArray."""
        # Configure pre-processing
        preprocessor = Preprocessor(
            fews_netcdf_kind=self.config.netcdf_kind,
            filter_stations=self.config.station_ids,
            filter_forecast_periods=self.config.forecast_periods.timedelta64,
        )

        # Observations
        if self.config.timeseries_kind == TimeseriesKind.observed_historical:
            with xr.open_mfdataset(
                self.config.paths,  # type:ignore[arg-type] # generator is acceptable argument
                preprocess=preprocessor,
            ) as dataset:
                dataset.load()

        # Simulations
        if self.config.netcdf_kind == FewsNetCDFKind.simulated_forecast_per_forecast_reference_time:
            with xr.open_mfdataset(
                self.config.paths,  # type:ignore[arg-type] # generator is acceptable argument
                combine="by_coords",
                preprocess=preprocessor,
                coords="minimal",
                compat="override",
            ) as dataset:
                # Load the dataset into memory
                #   for now, we assume the dataset with lead time filtering fits into memory
                dataset.load()

            # Transform forecast reference time simulation to internal datamodel based on
            #   forecast_period
            dataset = FewsNetCDF.transform_frt_simulation_to_internal_datamodel(
                dataset,
            )

        # Final transformation for all data
        data_array = FewsNetCDF.convert_to_data_array_and_set_variable_and_units_coords(
            dataset,
            source=self.config.source,
            timeseries_kind=self.config.timeseries_kind,
        )

        # Select only relevant time steps, given the configured verification_period
        self.data_array = data_array.sel(
            {  # type:ignore[misc]
                StandardDim.time: slice(
                    self.config.verification_period.start,
                    self.config.verification_period.end,
                ),
            },
        )

        return self
