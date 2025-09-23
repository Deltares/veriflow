"""Module for tests of datasinks."""

from pathlib import Path

from dpyverification.configuration.default.scores import CrpsForEnsembleConfig
from dpyverification.datamodel.main import OutputDataset, SimObsDataset
from dpyverification.datasinks.cf_compliant_netdf import CFCompliantNetCDF
from dpyverification.scores.probabilistic import CrpsForEnsemble


def test_write_data_cf_compliant_netcdf_no_scores(
    tmp_path: Path,
    simobsdataset_fews_netcdf_data: SimObsDataset,
    datasink_cf_compliant_netcdf: CFCompliantNetCDF,
) -> None:
    """Test writing data in cf-compliant NetCDF."""
    # Initialize output dataset
    output_dataset = OutputDataset(
        obs=simobsdataset_fews_netcdf_data.obs,
        sim=simobsdataset_fews_netcdf_data.sim,
    )

    # Write data from the output dataset
    datasink_cf_compliant_netcdf.write_data(output_dataset.get_output_dataset())
    assert (tmp_path / "test.nc").exists()


def test_write_data_cf_compliant_netcdf_crps(
    tmp_path: Path,
    simobsdataset_fews_netcdf_data: SimObsDataset,
    score_config_crps: CrpsForEnsembleConfig,
    datasink_cf_compliant_netcdf: CFCompliantNetCDF,
) -> None:
    """Test writing data in cf-compliant NetCDF."""
    # Initialize output dataset
    output_dataset = OutputDataset(
        obs=simobsdataset_fews_netcdf_data.obs,
        sim=simobsdataset_fews_netcdf_data.sim,
    )

    # Add a crps computation to the output dataset
    score = CrpsForEnsemble(score_config_crps)
    crps_result = score.compute(
        data=simobsdataset_fews_netcdf_data,
    )
    # Write data from the output dataset
    output_dataset.add_score(score.kind, crps_result)

    # Write the data
    datasink_cf_compliant_netcdf.write_data(output_dataset.get_output_dataset())
    assert (tmp_path / "test.nc").exists()
