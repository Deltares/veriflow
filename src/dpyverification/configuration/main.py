"""Classes to generate a valid configuration object from the specifcation in a file."""

import pathlib
from enum import Enum

import yaml
from jsonschema import validate

_schema = {
    "type": "object",
    "properties": {
        "fileversion": {"type": "string"},
    },
}


class ConfigTypes(Enum):
    """The types of configuration files that are supported."""

    YAML = "yaml"
    """ A yaml / json file. Actually only a subset of yaml, since will use jsonschema to validate,
     but should be ok for our purposes"""

    RUNINFO = "runinfo"
    """ FEWS general adapter runinfo file """


class Config:
    """The configuration definition of the dpyverification pipeline."""

    def __init__(self, configfile: pathlib.Path, configtype: ConfigTypes) -> None:
        if configtype is ConfigTypes.RUNINFO:
            # parse the runinfo into a yaml
            yamlcontent = {"fileversion": "0.0.1"}
        elif configtype is ConfigTypes.YAML:
            with configfile.open() as cf:
                yamlcontent = yaml.safe_load(cf)

        # conversion from older fileversion to current schema
        # NOT IMPLEMENTED YET, because we have not had a fileversion update

        # check the yaml
        validate(yamlcontent, _schema)

        self.filename = configfile
        self.configtype = configtype
