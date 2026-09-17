"""The base configuration definitions for the veriflow pipeline."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Self

import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
from pydantic.json_schema import SkipJsonSchema

from veriflow.cache.config import ZarrCacheConfig
from veriflow.constants import DataType, SpatialType, StandardDim

from .utils import CRSString, LeadTimes, Source, TimePeriod, VerificationPair, VerificationPeriod

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    # "BaseConfig",
    # "BaseDatasourceConfig",
    # "BaseDatasinkConfig",
    # "BaseScoreConfig",
    # "BaseCategoricalScoreConfig",
    "GeneralInfoConfig",
    "IdMap",
    "IdMappingConfig",
]


class GeneralInfoConfig(BaseModel):
    """General configuration information that is shared across the pipeline."""

    verification_period: Annotated[
        VerificationPeriod,
        Field(description="The start and end of the verification period."),
    ]
    verification_pairs: Annotated[
        list[VerificationPair],
        Field(
            description="Specify pairs for computation of verification metrics. This allows you to "
            "verify multiple variables and multiple sources. For example, by specifying two pairs: "
            "verify simulated discharge for ModelA and ModelB against observed discharge from"
            "source Observed.",
        ),
    ]
    lead_times: Annotated[
        LeadTimes | None,
        Field(
            "A set of lead times for which to evaluate of the verification scores. "
            "A lead time is the timedelta between the forecast reference time of a forecast "
            "(t0, analysis_time, initialization time) and the valid time (time, observed time) "
            "and is also known as: lead time or forecast horizon)",
        ),
    ] = None
    cache: Annotated[
        ZarrCacheConfig | None,
        Field(
            description=(
                "Veriflow has built-in support for caching data in a local or remote zarr archive "
                "store. This allows you to reuse data across multiple runs of the pipeline, which "
                "can speed up execution. See the docs for more details and configuration options: "
                "https://deltares.github.io/veriflow/"
            ),
        ),
    ] = None

    def get_verification_pair(self, pair_id: str) -> VerificationPair:
        """Get one verification_pair by its id."""
        for pair in self.verification_pairs:
            if pair.id == pair_id:
                return pair
        # At runtime, the following statement should be unreachable, because
        #   we already validated all pair_ids are present during config initialization.
        msg = f"Pair with id '{pair_id}' not found in general verification_pairs configuration."
        raise ValueError(msg)

    @property
    def verification_period_on_time(self) -> TimePeriod:
        """The verification period along the time dimension."""
        if (
            self.verification_period.dimension == StandardDim.forecast_reference_time
            and self.lead_times is not None
        ):
            start = self.verification_period.start + self.lead_times.min
            end = self.verification_period.end + self.lead_times.max
            return TimePeriod(start=start, end=end)
        return self.verification_period

    @property
    def verification_period_on_frt(self) -> TimePeriod:
        """The verification period along the forecast reference time dimension."""
        if self.verification_period.dimension == "time" and self.lead_times is not None:
            start = self.verification_period.start - self.lead_times.max
            end = self.verification_period.end - self.lead_times.min
            return TimePeriod(start=start, end=end)
        return self.verification_period

    @model_validator(mode="after")
    def verification_period_and_lead_times_consistent(self) -> Self:
        """Check that the verification period and lead times are consistent."""
        if self.lead_times is None and self.verification_period.dimension != StandardDim.time:
            msg = (
                "When no lead times are provided, the verification period should be defined "
                "along the time dimension (verification_period.dimension should be 'time')."
            )
            raise ValueError(msg)
        return self


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def get_external_to_internal_mapping(self, source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        # Check that the source is defined in the IdMap
        if not any(source in inner for inner in self.root.values()):
            msg = f"No IdMapping found for source: {source}"
            raise ValueError(msg)

        return {v[source]: k for k, v in self.root.items()}

    def rename_external_to_internal(self, external_ids: set[str], source: str) -> set[str]:
        """Apply the mapping to a set of external IDs for a given source.

        Returns the corresponding internal IDs.
        """
        ext_to_int = self.get_external_to_internal_mapping(source)
        return {ext_to_int[ext] for ext in external_ids if ext in ext_to_int}

    @property
    def sources(self) -> set[str]:
        """Return the set of all sources defined in the IdMap."""
        return {source for inner in self.root.values() for source in inner}


class IdMappingConfig(BaseModel):
    """Config for mapping external ids to their internal definition."""

    variable: Annotated[
        IdMap | None,
        Field(
            description="Mapping of internal to external definitions per source as a dictionary. "
            "The key corresponds to the internal definition and the value is another dictionary "
            "with keys corresponding to the source and the value to the external definition. ",
        ),
    ] = None
    station: Annotated[
        IdMap | None,
        Field(
            description="Mapping of internal to external definitions per source as a dictionary. "
            "The key corresponds to the internal definition and the value is another dictionary "
            "with keys corresponding to the source and the value to the external definition. ",
        ),
    ] = None

    def apply(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply the configured id mapping to a dataset.

        Variable names (data variable names) and station identifiers are renamed from the
        external (source-specific) definition to the internal definition.
        """
        source = str(dataset.attrs.get("source_id", ""))  # type:ignore[misc]

        # Re-assign variable definitions, if mapping is provided for source
        if self.variable is not None:
            ext_to_int = self.variable.get_external_to_internal_mapping(source)
            # Restrict to variables that actually exist as data_vars on the dataset
            rename_map = {
                ext: internal for ext, internal in ext_to_int.items() if ext in dataset.data_vars
            }
            if len(rename_map) > 0:
                dataset = dataset.rename_vars(rename_map)

        # Re-assign station coordinates, if mapping is provided for source
        if self.station is not None:
            dataset = dataset.assign_coords(
                {  # type:ignore[misc]
                    StandardDim.station: (  # type:ignore[misc]
                        StandardDim.station,
                        dataset[StandardDim.station]  # type:ignore[misc]
                        .to_series()
                        .replace(self.station.get_external_to_internal_mapping(source))
                        .to_numpy(),
                    ),
                },
            )
        return dataset


class BaseConfig(BaseModel):
    """A base config.

    Each element in the pipeline (datasource, score, datasink)
    inherits from this BaseConfig, so that each config has a 'kind' attribute.

    Based on a user-input to the configuration field 'kind', the pipeline
    will find the correct user-provided class for either a Datasource, score
    or a Datasink.
    """

    # Accept additional fields.
    # This is a requirement to make sure that all fields are
    # available after initializing the config instance when
    # the fields are created by external users and thus
    # not known upfront.
    model_config = ConfigDict(extra="allow")


class BaseDatasourceConfig(BaseConfig):
    """
    Base config for a datasource config.

    Specific config definitions should inherit from
    this base class.
    """

    import_adapter: str
    source_id: Source
    data_type: DataType
    spatial_type: SpatialType = SpatialType.point
    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    id_mapping: SkipJsonSchema[IdMappingConfig] | None = None

    @property
    def lead_times(self) -> LeadTimes | None:
        return self.general.lead_times

    @property
    def verification_period(self) -> TimePeriod:
        return self.general.verification_period

    @property
    def verification_period_on_frt(self) -> TimePeriod:
        return self.general.verification_period_on_frt

    @property
    def verification_period_on_time(self) -> TimePeriod:
        return self.general.verification_period_on_time


class BaseDatasinkConfig(BaseConfig):
    """
    Base config for a datasink config.

    Specific config definitions should inherit from
    this base class.
    """

    export_adapter: str
    force_overwrite: Annotated[
        bool,
        Field(description="Whether to force overwrite existing output files."),
    ] = True
    include_output: Annotated[
        bool,
        Field(description="Whether to include output in the output."),
    ] = True
    include_aligned_input_data: Annotated[
        bool,
        Field(
            description="Whether to include aligned input data in the output (per verification "
            "pair). "
            "See for reference: "
            "https://deltares.github.io/veriflow/api/_generated/veriflow.datatree.datatree.html",
        ),
    ] = True
    include_input_data: Annotated[
        bool,
        Field(
            description="Whether to include the input data in the output. This "
            "is the raw and validated input data as provided by the datasource(s)."
            "See for reference: "
            "https://deltares.github.io/veriflow/api/_generated/veriflow.datatree.datatree.html",
        ),
    ] = False
    crs: Annotated[
        CRSString | None,
        Field(
            default=None,
            description="Optional coordinate reference system for the output. When set, the "
            "results' coordinates are reprojected to this CRS before writing."
            "See for reference: "
            "https://deltares.github.io/veriflow/api/_generated/veriflow.datatree.datatree.html",
        ),
    ] = None

    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    @property
    def verification_period(self) -> TimePeriod:
        return self.general.verification_period


class BaseScoreConfig(BaseConfig):
    """
    Base config for a score config.

    Specific config definitions should inherit from
    this base class.
    """

    score_adapter: str
    general: SkipJsonSchema[GeneralInfoConfig]  # Do not serialize to json schema, since general
    # config is propagated from the general config section in the main config. This will prevent
    # users that use the json-schema for making config having to explicitly set a duplicate general
    # configuration section for each datasource.

    crs: Annotated[
        CRSString | None,
        Field(
            default=None,
            description="Optional coordinate reference system for score computation. When set, "
            "obs and sim coordinates are reprojected to this CRS before computing the score and "
            "the results are expressed in it. When omitted, observations and simulations must "
            "share the same CRS.",
        ),
    ] = None

    verification_pair_ids: Annotated[
        list[str],
        Field(
            description="Optional field to select verification_pairs from the general "
            "configuration, by providing a list of verification pair ids from the general config. "
            "Only these pair ids will be used in the computation of this score.",
        ),
    ] = []

    @property
    def verification_pairs(self) -> list[VerificationPair]:
        """The configured variable pairs.

        If the verification_pairs element is configured for the score, filter only these ids
        from the verification_pairs defined in general config.
        """
        if self.verification_pair_ids == []:
            return self.general.verification_pairs
        return [
            self.general.get_verification_pair(pair_id) for pair_id in self.verification_pair_ids
        ]

    @property
    def lead_times(self) -> LeadTimes | None:
        return self.general.lead_times

    @model_validator(mode="after")
    def verification_pair_ids_valid(self) -> Self:
        """Check provided filter for verification pairs contains valid ids."""
        valid_pair_ids: Generator[str, None, None] = (
            pair.id for pair in self.general.verification_pairs
        )

        for pair_id in self.verification_pair_ids:
            if pair_id not in valid_pair_ids:
                msg = (
                    f"Pair id '{pair_id}' in filter_verification_pairs is not present in "
                    "the general configuration for verification_pairs. "
                    "Please make sure ids match exactly."
                )
                raise ValueError(msg)
        return self


class BaseEvent(BaseModel):
    """Base class for event definitions."""


class BaseCategoricalScoreConfig(BaseScoreConfig):
    """
    Base config for a categorical score config.

    Specific config definitions should inherit from
    this base class.
    """

    events: Annotated[
        Iterable[
            BaseEvent
        ],  # we use Iterable instead of list to also allow subclasses of BaseEvent (see: https://docs.python.org/3/library/typing.html#generics)
        Field(
            description="A list of event definitions. For each event, a categorical score will be "
            "computed.",
        ),
    ]
