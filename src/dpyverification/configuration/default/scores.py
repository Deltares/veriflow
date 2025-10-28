"""A module for default implementation of scores."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, RootModel

from dpyverification.configuration.base import BaseScoreConfig
from dpyverification.constants import ScoreKind, StandardDim, SupportedContinuousScore


class ComputableDim(StrEnum):
    """List of dimensions that can be used for preservation or reduction in computations."""

    time = "time"
    station = "station"
    forecast_period = "forecast_period"


class ReduceDims(RootModel[list[ComputableDim]]):
    """A list of dimensions over which to compute a score."""

    @property
    def values(self) -> list[ComputableDim]:
        """Return a list of values."""
        return self.root

    @property
    def inverse(self) -> list[StandardDim]:
        """Get the preservable dims as inverse of reduce dims."""
        return [
            dim
            for dim in [
                StandardDim.variable,
                StandardDim.time,
                StandardDim.station,
                StandardDim.forecast_period,
            ]
            if dim not in self.root
        ]


ReduceDimsWithDefault = Annotated[
    ReduceDims,
    Field(default_factory=lambda: ReduceDims([ComputableDim.time])),
]


class IdMap(RootModel[dict[str, dict[str, str]]]):
    """Mapping from internal IDs to external IDs per data source."""

    def get_external_to_internal_mapping(self, data_source: str) -> dict[str, str]:
        """Return external → internal mapping for this data source."""
        return {v[data_source]: k for k, v in self.root.items()}


class RankHistogramConfig(BaseScoreConfig):
    """A rank histogram config element."""

    kind: Literal[ScoreKind.rank_histogram]
    reduce_dims: ReduceDimsWithDefault


class CrpsForEnsembleConfig(BaseScoreConfig):
    """Configuration for CRPS for ensemble.

    For reference, see: See: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_for_ensemble
    """

    kind: Literal[ScoreKind.crps_for_ensemble]
    method: Annotated[
        Literal["ecdf", "fair"],
        Field(
            description=(
                "Method to compute the cumulative distribution function from an ensemble."
            ),
            default="ecdf",
        ),
    ]
    reduce_dims: ReduceDimsWithDefault


class CrpsCDFConfig(BaseScoreConfig):
    """Configuration for CRPS for CDF.

    For reference, see: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_cdf
    """

    kind: Literal[ScoreKind.crps_cdf]
    reduce_dims: ReduceDimsWithDefault
    integration_method: Annotated[
        Literal["exact", "trapz"],
        Field(
            description="The method of integration. 'exact' computes the exact integral, "
            "'trapz' uses a trapezoidal rule and is an approximation of the CRPS.",
        ),
    ] = "exact"


class ContinuousScoresConfig(BaseScoreConfig):
    """Configure multiple continuous scores."""

    kind: Literal[ScoreKind.continuous_scores]
    reduce_dims: ReduceDimsWithDefault
    scores: list[SupportedContinuousScore]
