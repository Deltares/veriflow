"""Command-line interface for running the verification pipeline."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

from dpyverification.configuration import ConfigFile
from dpyverification.configuration.file import ConfigKind
from dpyverification.constants import NAME
from dpyverification.pipeline import run_pipeline

logger = logging.getLogger(__name__)


app = typer.Typer(  # The main app for the command-line interface
    help=(
        f"Welcome to the {NAME} command line interface. The interface allows you to run a "
        "verification pipeline."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},  # type:ignore[misc]
    no_args_is_help=True,
)
run_pipeline_subcommand = typer.Typer()  # Add subcommand group


@app.callback()  # type:ignore[misc]
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:  # noqa: FBT002
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
    set_verification_period_start: Annotated[
        datetime | None,
        typer.Option(
            help="Override verification period start (ISO format: YYYY-MM-DDTHH:MM:SS)",
        ),
    ] = None,
    set_verification_period_end: Annotated[
        datetime | None,
        typer.Option(
            help="Override verification period end (ISO format: YYYY-MM-DDTHH:MM:SS)",
        ),
    ] = None,
) -> None:
    """Run the verification pipeline from the command line."""
    logger.info("Reading and validating config file.")
    # Load the configuration
    #   at this point, Typer has already validate path is an existing, readable file
    config = ConfigFile(
        config_file=path_to_yaml_config,
        config_type=ConfigKind.YAML,
    ).content

    # Handle optional overrides
    if set_verification_period_start is not None:
        config.general.verification_period.start = set_verification_period_start
    if set_verification_period_end is not None:
        config.general.verification_period.end = set_verification_period_end

    # Run the pipeline
    run_pipeline(config=config)


app.add_typer(run_pipeline_subcommand)


if __name__ == "__main__":
    app()
