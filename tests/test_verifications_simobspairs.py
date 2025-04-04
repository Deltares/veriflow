"""Test the pixml module of the dpyverification.datasources package."""

import numpy as np
import pandas as pd
import xarray as xr
from dpyverification.configuration.schema import SimObsVariables
from dpyverification.constants import DataModelCoords
from dpyverification.verifications.simobspairs import _simobs

# mypy: disable-error-code="misc"


def test_simobs_output() -> None:
    """Check the output of the simobs calculation.

    Are the values from the correct datetimes filtered, and placed at the correct output datetimes?
    """
    leadtimes = [np.timedelta64(1, "h"), np.timedelta64(2, "h")]
    varnames = ["obsvar1", "simvar1", "obsvar2", "simvar2"]
    variablepairs = [
        SimObsVariables(obs=varnames[0], sim=varnames[1]),
        SimObsVariables(obs=varnames[2], sim=varnames[3]),
    ]
    first_time = np.datetime64("2014-09-06") + np.timedelta64(1, "h")
    time = pd.date_range(first_time, periods=3, freq="h")
    simstarts = pd.date_range("2014-09-06", periods=2, freq="h")
    x = np.array([2, 3, 4])
    y = [x * 2, (x + 4) * 2]
    x1 = [x * 3, x * 4]
    y1 = [
        [x * 5, (x + 4) * 6],
        [x * 7, (x + 4) * 8],
    ]
    extradim = "extradim"
    # Note the order of the dimensions of the variables, and how that compares to the order of the
    #   dimensions of the output variables, when reasoning about where the values in the output
    #   array should be located.
    input_dataset = xr.Dataset(
        data_vars={
            varnames[0]: (
                DataModelCoords.time.name,
                x,
            ),
            varnames[1]: (
                [DataModelCoords.leadtime.name, DataModelCoords.time.name],
                y,
            ),
            varnames[2]: (
                [extradim, DataModelCoords.time.name],
                x1,
            ),
            varnames[3]: (
                [extradim, DataModelCoords.leadtime.name, DataModelCoords.time.name],
                y1,
            ),
        },
        coords={
            DataModelCoords.simstart.name: simstarts,
            DataModelCoords.leadtime.name: leadtimes,
            DataModelCoords.time.name: time,
            extradim: [1, 2],
        },
    )

    output = _simobs(input_dataset, leadtimes, variablepairs)

    # The time dimension of the output is expected to be equal to the input time
    assert np.array_equal(
        output[DataModelCoords.time.name].data,
        time.values,
    )
    # Check the dims of the output variables are in the 'expected' order
    #   The order also determines how the values should be compared
    assert output["obsvar1_simobspair_obsvar1"].dims == ("time", "leadtime")
    assert output["obsvar1_simobspair_simvar1"].dims == ("leadtime", "time")
    assert output["obsvar2_simobspair_obsvar2"].dims == ("extradim", "time", "leadtime")
    assert output["obsvar2_simobspair_simvar2"].dims == ("extradim", "leadtime", "time")
    # Check the values of the output variables
    assert np.array_equal(
        output["obsvar1_simobspair_obsvar1"].data,
        [[2.0, np.nan], [3.0, 3.0], [np.nan, 4.0]],
        equal_nan=True,
    )
    assert np.array_equal(
        output["obsvar1_simobspair_simvar1"].data,
        [[4.0, 6.0, np.nan], [np.nan, 14.0, 16.0]],
        equal_nan=True,
    )
    assert np.array_equal(
        output["obsvar2_simobspair_obsvar2"].data,
        [
            [[6.0, np.nan], [9.0, 9.0], [np.nan, 12.0]],
            [[8.0, np.nan], [12.0, 12.0], [np.nan, 16.0]],
        ],
        equal_nan=True,
    )
    assert np.array_equal(
        output["obsvar2_simobspair_simvar2"].data,
        [
            [[10.0, 15.0, np.nan], [np.nan, 42.0, 48.0]],
            [[14.0, 21.0, np.nan], [np.nan, 56.0, 64.0]],
        ],
        equal_nan=True,
    )
