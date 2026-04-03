"""A module containing a generic base class for datasources, datasinks and scores."""

from abc import ABC
from typing import Any, Self

from dpyverification.configuration.base import BaseConfig


class Base(ABC):
    """Abstract base class for datasources, scores and datasinks.

    All definitions of datasources, scores and datasinks should inherit
    from the base class. It ensures classes have attributes 'kind' and
    'schema', that allow the pipeline to find the correct classes, based
    on a user-provided configuration for 'kind'. In addition, this base-class
    has a classmethod, that allows the pipeline to generate an instance of a
    subclass, based on a configuration.
    """

    kind: str
    config_class: type[BaseConfig]

    def __init_subclass__(cls) -> None:
        """Init subclass."""
        super().__init_subclass__()
        if not hasattr(cls, "kind"):
            msg = f"{cls.__name__} must define a 'kind' class variable."
            raise TypeError(msg)
        if not hasattr(cls, "config_class"):
            msg = f"{cls.__name__} must define a 'config_class' class variable."
            raise TypeError(msg)

    def __init__(self, config: BaseConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, raw_config: dict[str, Any]) -> Self:  # type: ignore  # noqa: PGH003
        """Initialize class from config dict."""
        config = cls.config_class(**raw_config)  # type: ignore[misc]
        return cls(config)
