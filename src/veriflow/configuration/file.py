"""Classes to generate a valid configuration object from the specification in a file."""

import pathlib
from enum import StrEnum, unique

import yaml

from .config import (
    Config,
)


@unique
class ConfigKind(StrEnum):
    """The types of configuration files that are supported."""

    YAML = "yaml"
    """ A yaml / json file"""

    RUNINFO = "runinfo"
    """ FEWS general adapter runinfo file """


class ConfigFile:
    """The configuration definition of the veriflow pipeline."""

    def __init__(
        self,
        config_file: pathlib.Path,
        config_type: ConfigKind | str,
    ) -> None:
        conftype = ConfigKind(
            config_type,
        )
        if conftype is ConfigKind.RUNINFO:
            # parse the runinfo into a yaml
            yamlcontent = {
                "fileversion": "0.0.1",
            }
            # TODO(AU): Implement parsing of a runinfo xml file to valid config dict # noqa: FIX002
            #   https://github.com/Deltares/veriflow/issues/8
        elif conftype is ConfigKind.YAML:
            with config_file.open() as cf:
                yamlcontent = yaml.safe_load(cf)
            # conversion from older fileversion to current schema
            # NOT IMPLEMENTED YET, because we have not had a fileversion update

        self.filename = config_file
        self.configtype = config_type

        # Propagate the general config to datasources
        for datasource in yamlcontent["datasources"]:
            datasource.update({"general": yamlcontent["general"]})  # type: ignore[attr-defined]
            if "id_mapping" in yamlcontent:
                datasource.update({"id_mapping": yamlcontent["id_mapping"]})  # type: ignore[attr-defined]

        # Propagate the general config to scores
        for score in yamlcontent["scores"]:
            score.update({"general": yamlcontent["general"]})  # type: ignore[attr-defined]

        # Propagate the general config to sinks
        if "datasinks" in yamlcontent:
            for sink in yamlcontent["datasinks"]:
                sink.update({"general": yamlcontent["general"]})  # type: ignore[attr-defined]

        self.content = Config(**yamlcontent)  # type: ignore[arg-type] # The derived type based on the hardcoded dict is not correct, but that is expected for now
