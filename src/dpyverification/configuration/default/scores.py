"""A module for default implementation of scores."""

from typing import Annotated, Literal

from pydantic import AfterValidator, Field

from dpyverification.configuration.base import BaseScoreConfig
from dpyverification.constants import DataModelDims, ScoreKind


class SimObsPairsConfig(BaseScoreConfig):
    """A sim obs pairs config element."""

    kind: Literal[ScoreKind.SIMOBSPAIRS]


class RankHistogramConfig(BaseScoreConfig):
    """A rank histogram config element."""

    kind: Literal[ScoreKind.RANKHISTOGRAM]
    reduce_dims: Annotated[
        list[DataModelDims] | None,
        Field(
            description=(
                "Dimension(s) over which to compute the histogram"
                "of ranks. Defaults to all dimensions."
            ),
        ),
    ] = None


class CrpsForEnsembleConfig(BaseScoreConfig):
    """A crps for ensemble config element."""

    @staticmethod
    def dim_is_not_ensemble(value: DataModelDims) -> DataModelDims:
        """Check dim is not ensemble dim."""
        if value == DataModelDims.ensemble:
            msg = "Cannot preserve ensemble dimension."
            raise ValueError(msg)
        return value

    kind: Literal[ScoreKind.CRPSFORENSEMBLE]
    method: Annotated[
        Literal["ecdf", "fair"],
        Field(
            description=(
                "Defaults to ecdf."
                "See: https://scores.readthedocs.io/en/stable/api.html#scores.probability.crps_for_ensemble"
            ),
            default="ecdf",
        ),
    ]
    preserve_dims: Annotated[
        list[DataModelDims] | None,
        AfterValidator(dim_is_not_ensemble),
        Field(
            description="List of dimension(s) to preserve in the output. Defaults to None.",
            default=None,
        ),
    ] = None
