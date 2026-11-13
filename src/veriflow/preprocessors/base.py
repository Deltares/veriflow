"""Base class for all preprocessors."""

import xarray as xr

from veriflow.base import Base
from veriflow.configuration.base import BasePreprocessorConfig


class BasePreprocessor(Base):
    """Base class for all preprocessors.

    A preprocessor transforms the input dataset of one source into a new dataset (e.g. aggregating
    to another time step). The result is staged in the ``input_data_processed`` node of the
    datatree and is used for pairing, while the raw dataset stays available in ``input_data``.
    """

    kind = ""  # to be defined by subclasses
    config_class: type[BasePreprocessorConfig] = BasePreprocessorConfig  # to be defined by
    # subclasses

    config: BasePreprocessorConfig

    def __init__(self, config: BasePreprocessorConfig) -> None:
        """Initialize the preprocessor with the given configuration."""
        super().__init__(config)

    def preprocess(self, dataset: xr.Dataset) -> xr.Dataset:
        """Preprocess a validated input dataset.

        The returned dataset must still be a valid input dataset: it keeps the dataset attributes
        (``data_type``, ``source_id``, ``crs``, ...) and the standard dimensions, so it can be
        paired and scored like the raw data.
        """
        raise NotImplementedError
