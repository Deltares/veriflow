"""The definition of the configuration settings.

This definition is used both as the schema for the configuration yaml file, and as the content of
the dpyverification configuration object.

To generate a yaml / json file with the json representation of this schema:

.. code-block:: python

    import pathlib
    import yaml
    from dpyverification.configuration import Config
    FILEPATH = pathlib.Path("YOUR_PATH_HERE")
    with FILEPATH.open("w") as myfile:
        yaml.dump(Config.model_json_schema(), myfile)
"""

# TODO(AU): Add pydantic Field with description, and maybe title, to all attributes. # noqa: FIX002
#   https://github.com/Deltares-research/DPyVerification/issues/9
#   Add pydantic Field with description, and maybe title, to approximately every attribute. To both
#   have a descriptive json schema when the json schema is generated from the pydantic objects, and
#   to document what the fields are for in the code. Maybe only for Literal attributes, the
#   description can be skipped. Do also add the description to private attributes, to document
#   their use.

import json
from collections.abc import Sequence
from functools import reduce
from pathlib import Path
from typing import Annotated, TypeVar

from pydantic import BaseModel, Field

from dpyverification.configuration.default.datasinks import (
    CFCompliantNetCDFConfig,
)
from dpyverification.configuration.default.datasources import (
    CsvConfig,
    FewsNetCDFConfig,
    FewsWebserviceConfig,
    NetCDFConfig,
)
from dpyverification.configuration.default.scores import (
    CategoricalScoresConfig,
    ContinuousScoresConfig,
    CrpsCDFConfig,
    CrpsForEnsembleConfig,
    RankHistogramConfig,
)

from .base import (
    BaseCategoricalScoreConfig,
    BaseDatasinkConfig,
    BaseDatasourceConfig,
    BaseScoreConfig,
    GeneralInfoConfig,
    IdMappingConfig,
)

TItem = TypeVar(
    "TItem",
    bound=BaseDatasourceConfig | BaseDatasinkConfig | BaseScoreConfig,
)


class Config(BaseModel):
    """Config object for running the verification pipeline."""

    fileversion: str
    general: GeneralInfoConfig
    datasources: Annotated[Sequence[BaseDatasourceConfig], Field(min_length=1)]
    scores: Annotated[Sequence[BaseScoreConfig | BaseCategoricalScoreConfig], Field(min_length=1)]
    datasinks: Annotated[Sequence[BaseDatasinkConfig] | None, Field(min_length=1)] = None
    id_mapping: IdMappingConfig | None = None

    @staticmethod
    def write_schema(
        output_path: Path,
        user_datasources_config: list[type[BaseDatasourceConfig]] | None = None,
        users_scores_config: list[type[BaseScoreConfig]] | None = None,
        user_datasinks_config: list[type[BaseDatasinkConfig]] | None = None,
    ) -> None:
        """Generate a YAML schema from the Pydantic model.

        By default, will write the schema for all default implementations of datasources, scores and
        datasinks. If you're using user-implementations of any of these, you can provide these
        classes to the function. This function will then add your user-implementation to the schema,
        together with the default implementations. As a result, YAML language support tools will be
        able to support both default and user-implementations. For example, by using the YAML
        extension in VSCode, using CTRL+SPACE will automatically suggest fields, based on the
        schema for both the default and user implementations.


        Parameters
        ----------
        output_path : Path
            Where to write the schema.
        user_datasources_config : list[type[BaseDatasourceConfig]] | None, optional
            Option to provide user-implemented config classes, by default None
        users_scores_config : list[type[BaseScoreConfig]] | None, optional
            Option to provide user-implemented config classes, by default None
        user_datasink_config : list[type[BaseDatasinkConfig]] | None, optional
            Option to provide user-implemented config classes, by default None

        """
        default_datasources_config = [
            FewsNetCDFConfig,
            FewsWebserviceConfig,
            CsvConfig,
            NetCDFConfig,
        ]
        default_scores_config = [
            CrpsForEnsembleConfig,
            RankHistogramConfig,
            CrpsCDFConfig,
            ContinuousScoresConfig,
            CategoricalScoresConfig,
        ]
        default_datasinks_config = [CFCompliantNetCDFConfig]

        def create_config_union(
            models: list[type[TItem]],
            discriminator: str,
        ) -> type:
            union_type = reduce(lambda a, b: a | b, models)  # type:ignore[misc, return-value, arg-type]
            return list[Annotated[union_type, Field(discriminator=discriminator)]]  # type:ignore[valid-type]

        merged_datasource_models = (
            default_datasources_config + user_datasources_config
            if user_datasources_config is not None
            else default_datasources_config
        )

        merged_scores_models = (
            default_scores_config + users_scores_config
            if users_scores_config is not None
            else default_scores_config
        )
        merged_datasinks_models = (
            default_datasinks_config + user_datasinks_config
            if user_datasinks_config is not None
            else default_datasinks_config
        )

        CombinedDataSourceConfig = create_config_union(  # noqa: N806
            merged_datasource_models,  # type:ignore[arg-type]
            discriminator="import_adapter",
        )
        CombinedScoreConfig = create_config_union(  # noqa: N806
            merged_scores_models,  # type:ignore[arg-type]
            discriminator="score_adapter",
        )
        CombinedDatasinkConfig = create_config_union(  # noqa: N806
            merged_datasinks_models,  # type:ignore[arg-type]
            discriminator="export_adapter",
        )

        class ConfigSchema(Config):
            datasources: CombinedDataSourceConfig  # type:ignore[valid-type]
            scores: CombinedScoreConfig  # type:ignore[valid-type]
            datasinks: CombinedDatasinkConfig | None = None  # type:ignore[valid-type]

        schema = ConfigSchema.model_json_schema()  # type:ignore[misc]

        # Write with explicit LF line endings so schema diffs are OS-independent.
        with output_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)  # type:ignore[misc]
            f.write("\n")
