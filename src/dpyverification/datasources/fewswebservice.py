"""Module for reading from and writing to a fews webservice."""

import tempfile
import zipfile
from pathlib import Path
from typing import Self

import xarray as xr

from dpyverification.api.fewswebservice import FewsWebserviceClient
from dpyverification.configuration import (
    FewsWebserviceInputConfig,
)
from dpyverification.constants import SimObsKind
from dpyverification.datasources.base import BaseDatasource
from dpyverification.datasources.fewsnetcdf import FewsNetcdfFile


class FewsWebservice(BaseDatasource):
    """For downloading data using a Delft-FEWS webservice."""

    # TODO(AU): Fix and document timezone information in fewswebservice requests # noqa: FIX002
    #   A hardcoded Z is added at the end, that cannot be right? See the issue for details:
    #   https://github.com/Deltares-research/DPyVerification/issues/43
    #

    kind = "fewswebservice"
    config_class = FewsWebserviceInputConfig
    # Annotate the correct type, otherwise mypy will infer from baseclass
    config: FewsWebserviceInputConfig
    # The datetime format that is used to pass datetimes to the fewswebservice
    datetime_format = "%Y-%m-%dT%H:%M:%SZ"
    timeout = 30

    def __init__(self, config: FewsWebserviceInputConfig) -> None:
        self.config = config
        self.simobstype = config.simobstype
        self.xarray = xr.Dataset()

        # Initialize the webservice client
        self.client = FewsWebserviceClient(
            url=self.config.auth_config.url.unicode_string(),
            username=self.config.auth_config.username.get_secret_value(),
            password=self.config.auth_config.password.get_secret_value(),
        )

    def get_data(self) -> Self:
        """Retrieve :py::class`~xarray.Dataset` from Delft-FEWS Webservice."""
        # Get observations
        if self.config.simobstype == SimObsKind.OBS:
            response = self.client.get_timeseries(
                location_ids=self.config.location_ids,
                parameter_ids=self.config.parameter_ids,
                module_instance_ids=self.config.module_instance_ids,
                qualifier_ids=self.config.qualifier_ids,
                start_time=self.config.general.verificationperiod.start,
                end_time=self.config.general.verificationperiod.end,
            )

            # Check response
            response.raise_for_status()

            # Write zip reponse to tmpfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(response.content)

            # Unzip tmpfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                zip_ref = zipfile.ZipFile(temp_file_path, "r")
                zip_ref.extractall(tmpdir)
                file_list = list(tmpdir_path.rglob("*nc"))

                # Check only one file is contained within the tmpfolder
                if len(file_list) != 1:
                    msg = "Unzipping obs file, yielde multiple .nc files. Expected exactly one."
                    raise ValueError(msg)

                tmp_nc_file_path = file_list[0]

                # Open xarray dataset and assign to self
                self.xarray = FewsNetcdfFile.nc_to_xarray(
                    Path(tmp_nc_file_path),
                    self.config.simobstype,
                )

        # Get simulations
        elif self.config.simobstype == SimObsKind.SIM:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Implement forecast retrieval, once Delft-FEWS development is completed.
                _ = tmpdir
                msg = "Simulations are not yet supported."
            raise NotImplementedError(msg)
        else:
            msg = (
                f"Simobstype {self.simobstype} not implemented yet. Only sim and obs are supported."
            )
            raise NotImplementedError(msg)
        return self
