"""PI-XML support module."""

from pathlib import Path
from typing import TYPE_CHECKING

import xarray as xr
from fewsio.pi import Timeseries  # type: ignore[import-untyped]

from dpyverification.datasources import GenericDatasource

if TYPE_CHECKING:
    import pandas as pd

# Ignore import untyped on fewsio since no type stub available. Unfortunately, this also means
#  that almost all locations where types from fewsio are used, need to have a type: ignore[misc],
#  because those types are seen as Any


class PiXmlFile(GenericDatasource):
    """For reading data from a pixml file."""

    @staticmethod
    def _pi_xml_to_xarray(path: Path) -> xr.DataArray:
        """Read pi-xml file and return xr.DataArray.

        Compatible with both observations and (ensemble) forecasts.

        Parameters
        ----------
        path : Path
            Path to the pi-xml file
        kind : Literal["sim", "obs"]
            String indicating the kind. Should be either sim (for simulations)
             or obs (for observations).


        Returns
        -------
        xr.DataArray
            DataArray representation of the pi-xml file.

        Raises
        ------
        TypeError
            Raised when pd.DataFrame.to_xarray() does not return xr.DataArray.
        """
        # Ignore import untyped on fewsio since no type stub available. Unfortunately, this also
        #  means that almost all locations where types from fewsio are used, need to have a
        #  type ignore[misc], because those types are seen as Any
        pi_df: pd.DataFrame

        # Load  pi-xml file
        pi_series: Timeseries = Timeseries(path, binary=False)  # type: ignore[misc]
        pi_df = pi_series.to_dataframe(ensemble_member=None).dropna(how="all")  # type: ignore[misc]
        pi_df.index.name = "time"  # type: ignore[misc]
        # Use the multi-index on the columns, convert to a multi-index on the rows. This will then
        #  be converted to Coordinates on the xarray.
        # Do not use future_stack=True with the stack() call, somehow certain values are then
        #  replaced with NaN (pandas bug?)
        pi_df2: pd.Series[float] = pi_df.stack(  # type: ignore[assignment] # noqa: PD013 We explicitly want to use the dependence on MultiIndex
            level=list(range(len(pi_df.columns.names))),
        )

        # Create xarray DataArray
        return pi_df2.to_xarray()

    @staticmethod
    def get_data(dsconfig: dict[str, str]) -> xr.DataArray:
        """Retrieve pixml content as an xarray DataArray."""
        filepath = Path(dsconfig["directory"]) / dsconfig["filename"]
        return PiXmlFile._pi_xml_to_xarray(filepath)
