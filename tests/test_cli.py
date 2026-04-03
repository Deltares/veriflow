"""Tests for the CLI entry point."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dpyverification.cli import app

runner = CliRunner()


def test_cli_run_without_overrides(
    cli_dummy_pipeline_config_yaml: Path,
) -> None:
    """Test running the CLI without overrides."""
    # When this exception is raised, we have successfully started a pipeline. Because we are using
    #   an invalid datasource, the pipeline will crash on start-up, but config is valid.
    with pytest.raises(ValueError, match="No item with type threshold_csv"):
        runner.invoke(
            app,
            [
                "run-pipeline",
                str(cli_dummy_pipeline_config_yaml),
            ],
            catch_exceptions=False,
        )


@pytest.mark.parametrize(
    ("cli_option_key", "cli_option_value"),
    [
        ("--set-verification-period-start", "2026-01-01T00:00:00"),
        ("--set-verification-period-end", "2027-01-01T00:00:00"),
    ],
    ids=[
        "--set-verification-period-start",
        "--set-verification-period-end",
    ],
)
def test_cli_run_with_valid_overrides(
    cli_dummy_pipeline_config_yaml: Path,
    cli_option_key: str,
    cli_option_value: str,
) -> None:
    """Test running the CLI with overrides."""
    # When this exception is raised, we have successfully started a pipeline. Because we are using
    #   an invalid datasource, the pipeline will crash on start-up, but config is valid.
    with pytest.raises(ValueError, match="No item with type threshold_csv"):
        runner.invoke(
            app,
            [
                "run-pipeline",
                str(cli_dummy_pipeline_config_yaml),
                cli_option_key,
                cli_option_value,
            ],
            catch_exceptions=False,
        )


@pytest.mark.parametrize(
    ("cli_option_key", "cli_option_value"),
    [
        ("--set-verification-period-start", "1"),
    ],
    ids=[
        "--set-verification-period-start",
    ],
)
def test_cli_run_with_invalid_overrides(
    cli_dummy_pipeline_config_yaml: Path,
    cli_option_key: str,
    cli_option_value: str,
) -> None:
    """Test running the CLI with invalid overrides."""
    expected_exit_code = 2
    result = runner.invoke(
        app,
        [
            "run-pipeline",
            str(cli_dummy_pipeline_config_yaml),
            cli_option_key,
            cli_option_value,
        ],
        catch_exceptions=True,
    )

    assert result.exit_code == expected_exit_code
    assert "--set-verification-period-start': '1' does not match the" in result.output
