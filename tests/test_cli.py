"""Tests for the CLI entry point."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dpyverification.cli import app
from dpyverification.configuration import Config

# mypy: disable-error-code=misc

runner = CliRunner()


test_cases = [
    {
        "cli_option_key": None,
        "cli_option_value": None,
        "expected_verification_period_start": "2026-01-01T00:00:00",
        "expected_verification_period_end": "2026-01-02T00:00:00",
    },
    {
        "cli_option_key": "--verification-period-start",
        "cli_option_value": "2025-01-01T00:00:00",
        "expected_verification_period_start": "2025-01-01T00:00:00",
        "expected_verification_period_end": "2026-01-02T00:00:00",
    },
    {
        "cli_option_key": "--verification-period-end",
        "cli_option_value": "2027-01-01T00:00:00",
        "expected_verification_period_start": "2026-01-01T00:00:00",
        "expected_verification_period_end": "2027-01-01T00:00:00",
    },
]


@pytest.mark.parametrize(
    "case",
    test_cases,
    ids=[
        "no-overrides",
        "override-verification-period-start",
        "override-verification-period-end",
    ],
)
def test_cli_run(
    cli_dummy_pipeline_config_yaml: Path,
    case: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test running the CLI with overrides."""
    captured_config: Config | None = None

    def fake_run_pipeline(*, config: Config) -> None:
        """Fake pipeline run function that just validates the config and raises an exception."""
        nonlocal captured_config
        captured_config = config

    # Patch the symbol used by the CLI module
    monkeypatch.setattr("dpyverification.cli.run_pipeline", fake_run_pipeline)

    if case["cli_option_key"] is None and case["cli_option_value"] is None:
        result = runner.invoke(
            app,
            [
                "run-pipeline",
                str(cli_dummy_pipeline_config_yaml),
            ],
            catch_exceptions=False,
        )
    else:
        result = runner.invoke(
            app,
            [
                "run-pipeline",
                str(cli_dummy_pipeline_config_yaml),
                case["cli_option_key"],
                case["cli_option_value"],
            ],
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    assert captured_config is not None
    assert (
        captured_config.general.verification_period.start.isoformat()
        == case["expected_verification_period_start"]
    )
    assert (
        captured_config.general.verification_period.end.isoformat()
        == case["expected_verification_period_end"]
    )


def test_cli_run_with_invalid_overrides(
    cli_dummy_pipeline_config_yaml: Path,
) -> None:
    """Test running the CLI with invalid overrides."""
    expected_exit_code = 2

    result = runner.invoke(
        app,
        [
            "run-pipeline",
            str(cli_dummy_pipeline_config_yaml),
            "--verification-period-start",
            "not-a-valid-datetime",
        ],
        catch_exceptions=False,
        color=False,
    )

    assert result.exit_code == expected_exit_code
