"""Module for tests of datasinks."""

from pathlib import Path

from dpyverification.configuration.default.scores import CrpsForEnsembleConfig
from dpyverification.datamodel.main import InputDataset, OutputDataset
from dpyverification.datasinks.cf_compliant_netcdf import CFCompliantNetCDF
from dpyverification.scores.probabilistic import CrpsForEnsemble


def test_write_data_cf_compliant_netcdf_no_scores(
    tmp_path: Path,
    input_dataset_fews_netcdf_simulated_forecast_ensemble: InputDataset,
    datasink_cf_compliant_netcdf: CFCompliantNetCDF,
) -> None:
    """Test writing data in cf-compliant NetCDF."""
    # Initialize output dataset
    output_dataset = OutputDataset(input_dataset_fews_netcdf_simulated_forecast_ensemble)

    for verification_pair in datasink_cf_compliant_netcdf.config.general.verification_pairs:
        # Write data from the output dataset
        fn = f"test_{verification_pair.id}"
        datasink_cf_compliant_netcdf.config.filename = fn
        datasink_cf_compliant_netcdf.write_data(
            output_dataset.get_output_dataset(verification_pair),
        )
        assert (tmp_path / fn).exists()


def test_write_data_cf_compliant_netcdf_crps(
    tmp_path: Path,
    input_dataset_fews_netcdf_simulated_forecast_ensemble: InputDataset,
    score_config_crps: CrpsForEnsembleConfig,
    datasink_cf_compliant_netcdf: CFCompliantNetCDF,
) -> None:
    """Test writing data in cf-compliant NetCDF."""
    # Initialize output dataset
    output_dataset = OutputDataset(input_dataset_fews_netcdf_simulated_forecast_ensemble)

    for verification_pair in score_config_crps.general.verification_pairs:
        # Add a crps computation to the output dataset
        score = CrpsForEnsemble(score_config_crps)
        obs, sim = input_dataset_fews_netcdf_simulated_forecast_ensemble.get_pair(verification_pair)
        crps_result = score.validate_and_compute(
            obs,
            sim,
        )
        # Write data from the output dataset
        output_dataset.add_score(score=crps_result, verification_pair_id=verification_pair.id)

        # Write the data
        fn = f"test_{verification_pair.id}"
        datasink_cf_compliant_netcdf.config.filename = fn
        datasink_cf_compliant_netcdf.write_data(
            output_dataset.get_output_dataset(verification_pair),
        )
        assert (tmp_path / fn).exists()
