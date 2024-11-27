"""Classes to generate a valid configuration object from the specification in a file."""

import pathlib
from enum import Enum

import yaml

from .schema import Calculation, ConfigSchema, DataSource, Output


class ConfigTypes(Enum):
    """The types of configuration files that are supported."""

    YAML = "yaml"
    """ A yaml / json file"""

    RUNINFO = "runinfo"
    """ FEWS general adapter runinfo file """


class Config:
    """The configuration definition of the dpyverification pipeline."""

    def __init__(self, configfile: pathlib.Path, configtype: ConfigTypes) -> None:
        if configtype is ConfigTypes.RUNINFO:
            # parse the runinfo into a yaml
            yamlcontent = {
                "fileversion": "0.0.1",
            }
            # TODO(AU): Implement parsing of a runinfo xml file to valid config dict # noqa: FIX002
            #   https://github.com/Deltares-research/DPyVerification/issues/8
        elif configtype is ConfigTypes.YAML:
            with configfile.open() as cf:
                yamlcontent = yaml.safe_load(cf)
            # conversion from older fileversion to current schema
            # NOT IMPLEMENTED YET, because we have not had a fileversion update

        # check the yaml and create python objects
        parsed_content = ConfigSchema(**yamlcontent)  # type: ignore[arg-type] # The derived type based on the hardcoded dict is not correct, but that is expected for now

        # TODO(AU): Check that specific leadtimes are subset of general leadtimes # noqa: FIX002
        #   https://github.com/Deltares-research/DPyVerification/issues/16
        #   Here, make this check part of the to-be-coded configuration validation

        self.filename = configfile
        self.configtype = configtype
        self.general = parsed_content.general
        self.datasources: list[DataSource] = parsed_content.datasources
        self.calculations: list[Calculation] = parsed_content.calculations
        self.output: list[Output] = parsed_content.output
