"""Read and write NetCDF files in a fews compatible format."""

from collections.abc import Iterator
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from veriflow.configuration.default.datasources import FewsNetCDFConfig, FewsNetCDFKind
from veriflow.constants import (
    FORECAST_DATA_TYPES,
    DataType,
    SpatialType,
    StandardCoord,
    StandardDim,
)
from veriflow.datasources.base import BaseDatasource
from veriflow.types import DataSpec

__all__ = [
    "FewsNetCDF",
    "FewsNetCDFConfig",
]

import logging

logger = logging.getLogger(__name__)


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
        filter_lead_times: list[np.timedelta64] | None = None,
    ) -> None:
        self.fews_netcdf_kind = fews_netcdf_kind
        self.variables = filter_variables
        self.stations = filter_stations
        self.lead_times = filter_lead_times

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
    def rename_to_internal(
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
        # Swap stations dim and drop stations dim.
        dataset = dataset.swap_dims(
            {FewsNetcdfDims.stations: StandardDim.station},  # type:ignore[misc]
        )

        if FewsNetcdfDims.stations in dataset:
            dataset = dataset.drop(FewsNetcdfDims.stations)

        # Only the case when retrieving full forecasts (per forecast reference time)
        if StandardCoord.station_name.name in dataset:
            dataset = dataset.set_coords(StandardCoord.station_name.name)

        # Rename analysis_time for simulations
        if FewsNetcdfDims.analysis_time in dataset:
            dataset = dataset.rename(
                {
                    FewsNetcdfDims.analysis_time: StandardDim.forecast_reference_time,  # type:ignore[misc]
                },
            )
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
    def set_internal_time_dims_on_forecast(
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Transform the FEWS NetCDF time dims/coords to internal."""
        lead_times = (
            (dataset[StandardDim.time] - dataset[StandardDim.forecast_reference_time])
            # type:ignore[misc]
            .to_numpy()
            .ravel()
        )
        ds = dataset.assign_coords(
            {StandardDim.lead_time: (StandardDim.time, lead_times)},  # type:ignore[misc]
        )

        ds = ds.swap_dims({StandardDim.time: StandardDim.lead_time}).drop_vars(  # type:ignore[misc]
            StandardDim.time,
        )

        # Re-compute time as 2d matrix along lead_time and forecast_reference_time
        time_index_2d = (
            (ds[StandardDim.forecast_reference_time] + ds[StandardDim.lead_time])
            # type:ignore[misc]
            .to_numpy()
            .swapaxes(0, 1)
        )

        # Now assign time
        ds = ds.assign_coords(
            {
                StandardDim.time: (  # type:ignore[misc]
                    (StandardDim.lead_time, StandardDim.forecast_reference_time),
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
        dataset = Preprocessor.rename_to_internal(
            dataset,
        )

        if self.fews_netcdf_kind == FewsNetCDFKind.simulated_historical:
            # We handle simulated historical NetCDFs separately, because this type is returned in
            #   a forecast format by the FEWS-Webservice, including the analysis_time dimensions.
            #   Internally, we want to convert this to a historical format, with time as the main
            #   time dimension. We assume for a given parameter_id, location_id and
            #   module_instance_id, there is only one analysis_time, which we convert to time.
            #   For reference, see: https://publicwiki.deltares.nl/spaces/FEWSDOC/pages/8683850/02+Data+Handling+in+Delft-FEWS#id-02DataHandlinginDelftFEWS-SimulatedHistoricaltimeseries
            dataset = dataset.drop_dims(StandardDim.forecast_reference_time)

        if self.fews_netcdf_kind == FewsNetCDFKind.simulated_forecast_per_forecast_reference_time:
            # Transform to full info sim dataset
            dataset = self.set_internal_time_dims_on_forecast(
                dataset,
            )

        # Filter variables
        if self.variables is not None:
            dataset = dataset[self.variables]

        # Sometimes time_bnds is included in the dataset, but this is not expected in the internal
        # datamodel and can cause issues with alignment later on, so we drop it if it exists.
        if "time_bnds" in dataset:
            dataset = dataset.drop_vars("time_bnds")

        # Filter stations
        if self.stations is not None:
            dataset = self.filter_stations(dataset, self.stations)

        # Filter lead times for simulations
        if (
            self.lead_times is not None
            and self.fews_netcdf_kind
            == FewsNetCDFKind.simulated_forecast_per_forecast_reference_time
        ):
            # Filter the relevant lead_times to maximize memory efficiency
            selector = {StandardDim.lead_time: self.lead_times}
            dataset = dataset.sel(selector)

        return dataset


def quantiles_to_cdf_data_array(
    sim: xr.DataArray,
    n_thresholds: int = 3000,
    padding_percentage: int = 5,
) -> xr.DataArray:
    """Create a cdf data array from Delft-FEWS NetCDF with quantiles.

    Verification metrics for CDFs may require a 'threshold' dimension to represent an array of
    thresholds of a continuous variable (like the crps_for_cdf function in the scores package).
    Here, the actual data variable represents probabilities of (non)-exceedance for the defined
    thresholds represented by the thresholds coordinate along the threshold dimension.

    In Delft-FEWS, however, such a datamodel is not available. Many users therefore use a standard
    FEWS NetCDF for ensembles, and use the realization dimension to represent probabilities (i.e.
    0.01, 0.02, ... 0.99) and the data variable (i.e. discharge) to represent the thresholds.

    This function converts a standard Delft-FEWS NetCDF to the internal input data structure.
    Because Delft-FEWS NetCDFs always only represent one forecast, the discretization of
    of probabilities and thresholds is always relative to the range of values found in one
    specific forecast (i.e. with discharges between 100-200). Because the input to this function
    is a set of FEWS NetCDFs, we need to resample the thresholds so that we can keep an accurate
    representation of each individual forecast CDF in a set of forecasts, but not completely blow
    up the matrix. Given the min and max value found in the dataset (with multiple forecasts),
    n_thresholds will be created by interpolating between min and max. For the set of forecasts, a
    new coordinate will then be created given the interpolated values. In this way, a new threshold
    dimension is created that can represent all values of the continuous variable found in the set
    of forecasts.

    Parameters
    ----------
    sim : xr.DataArray
        Forecast with realization dimension representing quantiles
    n_thresholds : int, optional
        Number of thresholds to use, by default 3000. Given the range of values found in the
        variable, interpolate between min-max with n_thresholds steps. For example if the range of
        discharge [m3/s] values found in the data is between 0-3000, the resolution of the threshold
        coordinate will correspond to 1 m3/s. Should be tuned according to the variable.
    padding_percentage : int, optional
        The % of padding to apply to the threshold coordinate, by default 5. This padding is applied
        so that the new threshold dimension (containing the variable values) has a wide enough range
        to also capture observed values during verification.


    Returns
    -------
    xr.DataArray
        A new data array with shared threshold dim and coords, compatible
        with the scores package (i.e. scores.probability.crps_cdf)
    """

    def check_non_decreasing_and_not_nan(arr: NDArray) -> None:
        """Check an array in non-decreasing and does not contain any NaN."""
        if np.isnan(arr).any():  # type:ignore[misc]
            msg = "NaN values found in input CDF."
            raise ValueError(msg)
        # Check for non-decreasing order
        if not (arr[:-1] <= arr[1:]).all():  # type:ignore[misc]
            msg = "Decreasing values found in input CDF."
            raise ValueError(msg)

    if StandardDim.realization not in sim.dims:
        msg = "No realization dimension found in input CDF."
        raise ValueError(msg)

    realization_index = sim[StandardDim.realization].to_numpy()  # type:ignore[misc]
    check_non_decreasing_and_not_nan(realization_index)  # type:ignore[misc]

    # Get the min / max probabilities
    min_probability: float = float(realization_index[0])  # type:ignore[misc]
    max_probability: float = float(realization_index[-1])  # type:ignore[misc]

    # Probabilities should be between 0-1. However, in FEWS NetCDFs, users may define
    #   a different scale, such as between 0-100. Scale any given array to the desired
    #   range, by finding base 10 logarithm and taking the ceiling, so we always scale by
    #   an integer.
    scaling_factor: int = 10 ** np.ceil(np.log10(max_probability))
    min_probability = min_probability / scaling_factor  # i.e. 99 > 0.99 and 0.99 > 0.99
    max_probability = max_probability / scaling_factor

    if not (0 <= min_probability <= max_probability <= 1):
        msg = "Probabilities must lie in [0, 1] after scaling."
        raise ValueError(msg)

    # Get the min / max values
    vmin = float(sim.min())
    vmax = float(sim.max())

    # Apply padding to min / max
    width = vmax - vmin
    padded_vmin = vmin - (padding_percentage * 0.01 * width)
    padded_vmax = vmax + (padding_percentage * 0.01 * width)

    # Define the steps and threshold index, for new shared coordinate
    thresholds = np.linspace(padded_vmin, padded_vmax, n_thresholds)

    def interpolate_cdf(cdf: NDArray[np.float64]) -> NDArray[np.float64]:
        # If all NaN, return a NaN array
        if np.all(np.isnan(cdf)):
            return np.full_like(thresholds, np.nan, dtype=np.float64)

        # If non all are Nan, require all not Nan and non-decreasing
        check_non_decreasing_and_not_nan(cdf)
        probs = np.linspace(min_probability, max_probability, len(cdf))
        return np.interp(
            thresholds,
            cdf,
            probs,
            left=0.0,
            right=1.0,
        )

    result: xr.DataArray = xr.apply_ufunc(
        interpolate_cdf,
        sim,
        input_core_dims=[["realization"]],  # type:ignore[misc]
        output_core_dims=[["threshold"]],  # type:ignore[misc]
        vectorize=True,
        dask="parallelized",
        output_sizes={"threshold": len(thresholds)},  # type:ignore[misc]
    )

    result = result.assign_coords(threshold=("threshold", thresholds))
    result.attrs.update(  # type:ignore[misc]
        {"data_type": DataType.simulated_forecast_probabilistic},  # type:ignore[misc]
    )
    result.name = sim.name

    return result


def parse_lead_time_netcdf_files(
    paths: Iterator[Path],
) -> xr.Dataset:
    """Parse NetCDF responses from get timeseries with leadTimes parameter."""

    def preprocess(dataset: xr.Dataset) -> xr.Dataset:
        """
        Preprocess individual files, set lead_time based on filename.

        When requesting data for a specific lead time via the FEWS-Webservice,
        the actual lead time used in the request is not available in the
        response. As a workaround, we prefix the filename with the lead time
        in milliseconds, access it via the dataset encoding and set it as a dim/coord
        on the dataset.
        """
        filename = Path(dataset.encoding["source"]).name  # type:ignore[misc]

        lead_time_millis = filename.split("_")[0]
        if not lead_time_millis.isalnum():
            msg = "Filename prefix is expected to be a numeric representing the lead time"
            "(lead time) in milliseconds. The provided prefix '{lead_time_millis}' is is not"
            "numeric and cannot be converted to a valid lead time."
            raise ValueError(msg)

        lead_time = np.timedelta64(int(lead_time_millis), "ms").astype(
            "timedelta64[ns]",
        )
        forecast_reference_times = dataset[StandardDim.time] - lead_time  # type:ignore[misc]

        # Set the station_name as coord instead of variable
        if FewsNetcdfCoord.station_names in dataset:
            dataset = dataset.set_coords(FewsNetcdfCoord.station_names)

        # Set forecast_reference_time dim / coord
        dataset = dataset.rename({StandardDim.time: StandardDim.forecast_reference_time})  # type:ignore[misc]
        dataset = dataset.assign_coords(
            {
                StandardDim.forecast_reference_time: (  # type:ignore[misc]
                    StandardDim.forecast_reference_time,
                    forecast_reference_times.to_numpy(),  # type:ignore[misc]
                ),
            },
        )
        dataset = dataset.sortby(StandardDim.forecast_reference_time)

        # Set coord (expected for alignment of individual arrays)
        dataset = dataset.assign_coords(
            {FewsNetcdfDims.stations: dataset[FewsNetcdfCoord.station_id].to_numpy()},  # type:ignore[misc]
        )
        # Assign lead_time as a dim/coord
        dataset = dataset.expand_dims(StandardDim.lead_time)
        return dataset.assign_coords(
            {
                StandardCoord.lead_time.name: (
                    StandardDim.lead_time,
                    [lead_time],  # type:ignore[misc]
                ),
            },
        )

    # Combine files in a deterministic order and make xarray's merge semantics explicit so the
    # result does not depend on platform-specific file iteration or library defaults.
    dataset_list = [preprocess(xr.open_dataset(path)) for path in paths]
    dataset = xr.merge(
        dataset_list,
        join="outer",
        compat="no_conflicts",
        combine_attrs="override",
    )

    dataset = dataset.sortby(StandardDim.forecast_reference_time)
    dataset = dataset.sortby(StandardDim.lead_time)

    # Decode byte-string coords
    dataset = Preprocessor.convert_byte_string_coord_to_utf8(
        dataset,
        coords=[FewsNetcdfCoord.station_id],
    )

    # Rename dims/coords to internal definitions
    dataset = Preprocessor.rename_to_internal(
        dataset,
    )

    # On resulting object, assign forecast_reference_time as coordinate
    return dataset.assign_coords(
        {  # type:ignore[misc]
            StandardDim.time: (  # type:ignore[misc]
                (StandardDim.forecast_reference_time, StandardDim.lead_time),
                (
                    dataset[StandardDim.forecast_reference_time] + dataset[StandardDim.lead_time]
                ).to_numpy(),  # type:ignore[misc]
            ),
        },
    )


class FewsNetCDF(BaseDatasource):
    """For reading data from a NetCDF file produced by Delft-FEWS."""

    kind = "fewsnetcdf"
    config_class = FewsNetCDFConfig
    supported_data_specs: ClassVar[set[DataSpec]] = {
        (DataType.observed_historical, SpatialType.point),
        (DataType.simulated_forecast_ensemble, SpatialType.point),
    }

    def __init__(self, config: FewsNetCDFConfig) -> None:
        self.config: FewsNetCDFConfig = config

    @property
    def configured_stations(self) -> set[str] | None:
        """Return the internal station identifiers configured for this datasource.

        This is needed for standardization of station identifiers across sources.
        """
        return set(self.config.station_ids) if self.config.station_ids is not None else None

    @configured_stations.setter
    def configured_stations(self, stations: set[str]) -> None:
        """Set the internal station identifiers configured for this datasource.

        This is needed for standardization of station identifiers across sources.
        """
        self.config.station_ids = list(stations)

    @property
    def configured_variables(self) -> set[str] | None:
        """Return the internal variable identifiers configured for this datasource.

        This is needed for standardization of variable identifiers across sources.
        """
        return set(self.config.parameter_ids) if self.config.parameter_ids is not None else None

    @configured_variables.setter
    def configured_variables(self, variables: set[str]) -> None:
        """Set the internal variable identifiers configured for this datasource.

        This is needed for standardization of variable identifiers across sources.
        """
        self.config.parameter_ids = list(variables)

    def standardize_dataset(
        self,
        dataset: xr.Dataset,
        data_type: DataType,
    ) -> xr.Dataset:
        """Standardize a fetched dataset to the internal datamodel.

        Ensures the station dim is indexed by the station id, sets standard dim ordering on each
        data variable, and ensures each data variable carries a ``units`` attribute (filling in
        ``"unknown"`` when missing).
        """
        # Set the station_id as index on station dim
        #   to ensure automatic alignment based on this coord later on.
        dataset = dataset.assign_coords(
            {
                StandardDim.station: dataset[StandardCoord.station.name].to_numpy(),  # type:ignore[misc]
            },
        )

        # Ensure each data var has a units attribute
        for var_name in dataset.data_vars:
            if "units" not in dataset[var_name].attrs:  # type:ignore[misc]
                dataset[var_name].attrs["units"] = "unknown"  # type:ignore[misc]

            # This is a bit hacky. When a user uses the FEWS-Webservice to retrieve data, and the
            # configured parameter_id contains a "." (e.g. "discharge.obs"), the returned NetCDF
            # will contain a data variable with the name "discharge_obs". This is because the "."
            # is not a valid character in NetCDF variable names according to the CF-conventions,
            # and FEWS auto-converts all variable names in NetCDF responses to "_".
            # This creates a diversion between a users configured parameter_id and the
            # actual data variable name in the NetCDF, which breaks our IdMapping logic applied
            # in the base class. As a workaround, when a user has configured an IdMapping logic
            # for a parameter containing a ".", we will back-convert the variable name by replacing
            # the "_" with a "." in the data variable name, so that it can be aligned with the
            # to match the configured parameter_id and the IdMapping.
            if not isinstance(var_name, str):
                msg = (
                    f"Data variable name is not a string: {var_name}. "
                    "This is unexpected and may indicate a problem with the NetCDF file."
                )
                raise TypeError(msg)

            if (
                self.config.id_mapping is not None
                and self.config.id_mapping.variable is not None
                and self.config.source in self.config.id_mapping.variable.sources
                and "_" in var_name
                and var_name.replace("_", ".")
                in list(
                    self.config.id_mapping.variable.get_external_to_internal_mapping(
                        self.config.source,
                    ),
                )
            ):
                dataset = dataset.rename(
                    {
                        var_name: var_name.replace("_", "."),  # type:ignore[misc]
                    },
                )

        # Standardize dim order per data variable
        if data_type in FORECAST_DATA_TYPES:
            transpose_dims = (
                StandardDim.station,
                StandardDim.forecast_reference_time,
                StandardDim.lead_time,
                ...,
            )
        else:
            # Historical simulations or observations
            transpose_dims = (StandardDim.station, StandardDim.time, ...)  # type:ignore[assignment]

        # Standardize dim order across all data variables.
        dataset = dataset.transpose(*transpose_dims)

        # Set the configured data type as a dataset-level attribute
        dataset.attrs["data_type"] = data_type  # type:ignore[misc]
        return dataset

    def fetch_data(self) -> Self:
        """Retrieve fewsnetcdf content as an xarray Dataset."""
        # Configure pre-processing
        preprocessor = Preprocessor(
            fews_netcdf_kind=self.config.netcdf_kind,
            filter_stations=self.config.station_ids,
            filter_variables=self.config.parameter_ids,
            filter_lead_times=self.config.lead_times.timedelta64
            if self.config.lead_times is not None
            else None,
        )

        # External historical data type
        if self.config.data_type in [DataType.observed_historical, DataType.simulated_historical]:
            dataset = xr.open_mfdataset(
                self.config.paths,  # type:ignore[arg-type] # generator is acceptable argument
                preprocess=preprocessor,
            )
        # Simulations - per forecast reference time
        if self.config.netcdf_kind == FewsNetCDFKind.simulated_forecast_per_forecast_reference_time:
            time_start = datetime.now()  # noqa: DTZ005
            dataset = xr.open_mfdataset(
                self.config.paths,  # type:ignore[arg-type] # generator is acceptable argument
                combine="by_coords",
                preprocess=preprocessor,
                coords="minimal",
                compat="override",
            )
            time_end = datetime.now()  # noqa: DTZ005
            msg = (
                f"Opened dataset for source '{self.config.source}' from {self.config.paths} "
                f"(took {(time_end - time_start).total_seconds():.2f} seconds)"
            )
            logger.info(msg)

        # Simulations - per lead time
        if self.config.netcdf_kind == FewsNetCDFKind.simulated_forecast_per_lead_time:
            dataset = parse_lead_time_netcdf_files(
                self.config.paths,
            )

        if self.config.data_type in FORECAST_DATA_TYPES:
            # After loading data into xr.Dataset, apply a filter on forecast reference time, based
            #   on the configured verification period
            dataset = dataset.sel(
                {
                    StandardDim.forecast_reference_time: slice(  # type:ignore[misc]
                        self.config.verification_period_on_frt.start,
                        self.config.verification_period_on_frt.end,
                    ),
                },
            )
        else:
            # Register the original start and end times from the dataset loaded from disk
            # Filter historical data (external_historical and simulated_historical)on time dim
            dataset = dataset.sel(
                {
                    StandardDim.time: slice(  # type:ignore[misc]
                        self.config.verification_period_on_time.start,
                        self.config.verification_period_on_time.end,
                    ),
                },
            )
            if dataset[StandardDim.time].size == 0:
                msg = (
                    "No time steps found in the dataset after applying the verification period "
                    f"filter. Dataset source: {self.config.source} no time steps between "
                    f"{self.config.verification_period_on_time.start} to "
                    f"{self.config.verification_period_on_time.end}"
                )
                raise ValueError(msg)

        # Load into memory, in the future support dask
        dataset.load()

        # Standardize to internal datamodel (dim order, units attr, dataset-level data_type)
        dataset = self.standardize_dataset(dataset, self.config.data_type)

        # For probabilistic data types, transform each data variable so that
        #   all cdf's share the same threshold dim
        if self.config.data_type == DataType.simulated_forecast_probabilistic:
            dataset = dataset.map(
                quantiles_to_cdf_data_array,
                keep_attrs=True,
            )

        # Assign to self
        self.dataset = dataset

        return self
