"""Classes to generate a valid configuration object from the specification in a file."""

import pathlib
from enum import StrEnum, unique

import yaml

from .schema import ConfigSchema, FewsWebserviceInput, GeneralInfo, SimObsPairs


@unique
class ConfigTypes(StrEnum):
    """The types of configuration files that are supported."""

    YAML = "yaml"
    """ A yaml / json file"""

    RUNINFO = "runinfo"
    """ FEWS general adapter runinfo file """


class Config:
    """The configuration definition of the dpyverification pipeline."""

    def __init__(self, configfile: pathlib.Path, configtype: ConfigTypes | str) -> None:
        conftype = ConfigTypes(
            configtype,
        )
        if conftype is ConfigTypes.RUNINFO:
            # parse the runinfo into a yaml
            yamlcontent = {
                "fileversion": "0.0.1",
            }
            # TODO(AU): Implement parsing of a runinfo xml file to valid config dict # noqa: FIX002
            #   https://github.com/Deltares-research/DPyVerification/issues/8
        elif conftype is ConfigTypes.YAML:
            with configfile.open() as cf:
                yamlcontent = yaml.safe_load(cf)
            # conversion from older fileversion to current schema
            # NOT IMPLEMENTED YET, because we have not had a fileversion update

        self.filename = configfile
        self.configtype = configtype
        self.content = ConfigSchema(**yamlcontent)  # type: ignore[arg-type] # The derived type based on the hardcoded dict is not correct, but that is expected for now
        self.validate_and_propagate()

    def validate_and_propagate(self) -> None:
        """
        Check and update the various parts of the config to be consistent, compatible and complete.

        - Propagate config settings to adjacent parts (i.e. from General to Datasources)
        - Propagate config settings to lower parts (i.e. from top-level object to objects in
          attributes)
        - Validate that various parts of the config are consistent / compatible with each other
          (i.e. General leadtimes with Datasource leadtimes)

        """
        # NOTE: for now, all in one method. When more validation added, have one function per
        #  top-level attribute?

        for datasource in self.content.datasources:
            if isinstance(datasource, FewsWebserviceInput):
                self._propagate_leadtimes(datasource, self.content.general, "FewsWebserviceInput")
                self._propagate_verificationperiod(
                    datasource,
                    self.content.general,
                    "FewsWebserviceInput",
                )

        for calculation in self.content.calculations:
            if isinstance(calculation, SimObsPairs):
                self._propagate_leadtimes(calculation, self.content.general, "SimObsPairs")

    @staticmethod
    def _propagate_leadtimes(
        specific: FewsWebserviceInput | SimObsPairs,
        general: GeneralInfo,
        name: str,
    ) -> None:
        if specific.leadtimes and general.leadtimes:
            missing = [
                dslead
                for dslead in specific.leadtimes.timedelta64
                if dslead not in general.leadtimes.timedelta64
            ]
            if any(missing):
                msg = (
                    f"The following leadtimes are used in {name}, but are not"
                    f" present in the general leadtimes: {missing}"
                )
                raise ValueError(msg)
        elif general.leadtimes:
            specific.leadtimes = general.leadtimes
        else:
            msg = (
                f"A {name} is used, but neither it nor the general section specifies the"
                f"leadtimes to be used."
            )
            raise ValueError(msg)

    @staticmethod
    def _propagate_verificationperiod(
        specific: FewsWebserviceInput,
        general: GeneralInfo,
        name: str,
    ) -> None:
        if specific.verificationperiod and general.verificationperiod:
            specific_period = specific.verificationperiod
            general_period = general.verificationperiod
            if (
                specific_period.start.datetime != general_period.start.datetime
                or specific_period.end.datetime != general_period.end.datetime
            ):
                msg = (
                    f"The verificationperiod used in {name} ({specific_period}) is not exactly"
                    f" equal to the verificationperiod in the general section"
                    f" ({general_period})."
                )
                raise ValueError(msg)
        elif general.verificationperiod:
            # NOTE: Do NOT use the X_period variables here, as this change needs to be applied
            #  to the `specific` object itself.
            specific.verificationperiod = general.verificationperiod
        else:
            msg = (
                f"A {name} is used, but neither it nor the general section specifies the"
                f"verificationperiod to be used."
            )
            raise ValueError(msg)
