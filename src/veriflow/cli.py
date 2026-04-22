"""Command-line interface for running the verification pipeline."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from veriflow.configuration import Config
from veriflow.configuration.file import ConfigFile
from veriflow.constants import NAME, VERSION
from veriflow.pipeline import run_pipeline

logger = logging.getLogger(__name__)


app = typer.Typer(
    help=(
        f"Welcome to the {NAME} command line interface. The interface allows you to run a "
        "verification pipeline."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},  # type:ignore[misc]
    no_args_is_help=True,
)
run_pipeline_subcommand = typer.Typer()


def _version_callback(*, value: bool) -> None:
    """Show the version and exit when --version/-V is provided."""
    if value:
        typer.echo(f"{NAME} version {VERSION}")
        raise typer.Exit


def override_general_info_config(
    config: Config,
    verification_period_start: datetime | None,
    verification_period_end: datetime | None,
) -> Config:
    """Override the general info config with command-line options."""
    if verification_period_start is None and verification_period_end is None:
        return config

    general_config = config.general.model_copy(deep=True)

    if verification_period_start is not None:
        logger.info(
            f"Overriding verification period start to {verification_period_start.isoformat()} "  # noqa: G004
            f"based on command-line input.",
        )
        general_config.verification_period.start = verification_period_start
    if verification_period_end is not None:
        logger.info(
            f"Overriding verification period end to {verification_period_end.isoformat()} "  # noqa: G004
            f"based on command-line input.",
        )
        general_config.verification_period.end = verification_period_end

    # Override the general config in the main config object.
    config.general = general_config

    # Because datasources, scores, and datasinks will have a copy of the general configuration,
    # we also need to override the verification period start for each of them
    if config.datasources is not None:
        for datasource in config.datasources:
            datasource.general = general_config
    if config.scores is not None:
        for score in config.scores:
            score.general = general_config
    if config.datasinks is not None:
        for datasink in config.datasinks:
            datasink.general = general_config

    logger.debug(
        "Successfully passed general info config overrides to all datasources, scores and "
        "datasinks.",
    )
    return config


@app.callback()  # type:ignore[misc]
def main(
    *,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    version: Annotated[  # noqa: ARG001
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Set the main callback for the app.

    We get the root logger and attach a RichHandler to it. In this way, log messages from the
    project (i.e. the pipeline) will be logged properly to the terminal.
    """
    logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    if not logger.hasHandlers():
        logger.addHandler(RichHandler())


@app.command("run-pipeline")  # type:ignore[misc]
def run_pipeline_cmd(
    path_to_yaml_config: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            readable=True,
            resolve_path=True,
            dir_okay=False,
            help="Path to the YAML configuration file.",
        ),
    ],
    verification_period_start: Annotated[
        datetime | None,
        typer.Option(
            help="Override verification period start (ISO format: YYYY-MM-DDTHH:MM:SS)",
        ),
    ] = None,
    verification_period_end: Annotated[
        datetime | None,
        typer.Option(
            help="Override verification period end (ISO format: YYYY-MM-DDTHH:MM:SS)",
        ),
    ] = None,
) -> None:
    """Run the verification pipeline from the command line."""
    # Load the YAML content from the provided path
    config = ConfigFile(config_file=path_to_yaml_config, config_type="yaml").content

    if verification_period_end is not None or verification_period_start is not None:
        # At least one of the verification period start and end is provided, so we need to override
        # the general info config with the provided values.
        config = override_general_info_config(
            config=config.model_copy(deep=True),
            verification_period_start=verification_period_start,
            verification_period_end=verification_period_end,
        )

    # Validate the configuration against the Pydantic model.
    logger.info("Validating configuration...")
    config = Config.model_validate(config)
    logger.info("Configuration validated successfully.")
    logger.info(
        "Starting the pipeline with the following verification period: %s to %s",
        verification_period_start,
        verification_period_end,
    )

    # Run the pipeline
    run_pipeline(config=config)


app.add_typer(run_pipeline_subcommand)


if __name__ == "__main__":
    app()
