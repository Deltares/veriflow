"""PI-XML support module."""

from pathlib import Path
from typing import TYPE_CHECKING, Self

import xarray as xr
from fewsio.pi import Timeseries  # type: ignore[import-untyped]

from dpyverification.configuration import DataSource, DataSourceTypeEnum, SimObsType
from dpyverification.datasources.genericdatasource import GenericDatasource

if TYPE_CHECKING:
    import pandas as pd

# Ignore import untyped on fewsio since no type stub available. Unfortunately, this also means
#  that almost all locations where types from fewsio are used, need to have a type: ignore[misc],
#  because those types are seen as Any


class PiXmlFile(GenericDatasource):
    """For reading data from a pixml file."""

    @staticmethod
    def _pi_xml_to_xarray(path: Path, kind: SimObsType) -> xr.Dataset:
        """Read pi-xml file and return xr.Dataset.

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
        xr.Dataset
            Dataset representation of the pi-xml file.

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

        # HARDCODED: Only for single parameter for now, need to think of how to handle multiple
        # Use parameter_id as variable name in the dataset
        par_ids: list[str] = (
            pi_df2.index.get_level_values("parameter_id").unique().to_numpy().tolist()  # type: ignore[misc]
        )
        if len(par_ids) != 1:
            raise NotImplementedError
        pi_df2 = pi_df2.droplevel("parameter_id")
        xname = par_ids[0]

        # WHAT TO DO WITH QUALIFIER_IDS? For now, check and squeeze. Need better definition of what
        #  it can be, and means, before using.
        qual_ids: list[frozenset[str]] = (
            pi_df2.index.get_level_values("qualifier_ids").unique().to_numpy().tolist()  # type: ignore[misc]
        )
        if len(qual_ids) != 1 or qual_ids[0] != frozenset():
            raise NotImplementedError
        pi_df2 = pi_df2.droplevel("qualifier_ids")

        if kind == SimObsType.obs and "ensemble_member" in pi_df2.index.names:  # type: ignore[misc]
            # obs should not contain ensemble member dimension, however fewsio Timeseries may have
            #  added it
            ens_ids: list[int] = (
                pi_df2.index.get_level_values("ensemble_member").unique().to_numpy().tolist()  # type: ignore[misc]
            )
            if len(ens_ids) != 1 or ens_ids[0] != 0:
                raise NotImplementedError
            pi_df2 = pi_df2.droplevel("ensemble_member")

        # Create xarray DataArray
        return pi_df2.to_xarray().to_dataset(name=xname)

    @classmethod
    def get_data(cls, dsconfig: DataSource) -> list[Self]:
        """Retrieve pixml content as an xarray DataArray."""
        if dsconfig.datasourcetype != DataSourceTypeEnum.pixml:
            msg = "Input dsconfig does not have datasourcetype pixml"
            raise TypeError(msg)
        if dsconfig.simobstype == SimObsType.combined:
            msg = "Cannot yet handle combined simobs data"
            raise NotImplementedError(msg)

        filepath = Path(dsconfig.directory) / dsconfig.filename
        pif = cls(dsconfig)
        pif.xarray = cls._pi_xml_to_xarray(filepath, dsconfig.simobstype)
        return [pif]

    # classmethod write_to_file remains explicitly not implemented for pi xml
