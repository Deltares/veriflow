"""Test the dpyverification.datamodel package."""

import numpy as np
import pandas as pd
import xarray as xr
from dpyverification.constants import DataModelCoords
from dpyverification.datamodel.main import DataModel

# mypy: disable-error-code="misc"

# TODO(AU): Add tests for the other methods of the datamodel.  # noqa: FIX002
#   Currently, only _create_intermediate_dataset is being tested. At the very least, a
#   dedicated issue with a list of what to test should be created as part of:
#   https://github.com/Deltares-research/DPyVerification/issues/38


def test_create_intermediate_dataset() -> None:
    """Check the output of _create_intermediate_dataset.

    Are the values from the correct datetimes filtered, and placed at the correct output datetimes?
    """
    varnames = ["obsvar1", "simvar1", "obsvar2", "simvar2"]
    time = pd.date_range("2014-09-06", periods=4, freq="h")
    simstarts = pd.date_range("2014-09-06", periods=2, freq="h")
    x = np.array([1, 2, 3, 4])
    y = [np.multiply(x, 2), np.multiply(np.add(x, 4), 2)]
    x1 = [np.multiply(x, 3), np.multiply(x, 4)]
    y1 = [
        [np.multiply(x, 5), np.multiply(np.add(x, 4), 6)],
        [np.multiply(x, 7), np.multiply(np.add(x, 4), 8)],
    ]
    extradim = "extradim"
    # Note the order of the dimensions of the variables, and how that compares to the order of the
    #   dimensions of the output variables, when reasoning about where the values in the output
    #   array should be located.
    inputdataset = xr.Dataset(
        data_vars={
            varnames[0]: (
                DataModelCoords.time.name,
                x,
            ),
            varnames[1]: (
                [DataModelCoords.simstart.name, DataModelCoords.time.name],
                y,
            ),
            varnames[2]: (
                [extradim, DataModelCoords.time.name],
                x1,
            ),
            varnames[3]: (
                [extradim, DataModelCoords.simstart.name, DataModelCoords.time.name],
                y1,
            ),
        },
        coords={
            DataModelCoords.simstart.name: simstarts,
            DataModelCoords.time.name: time,
            DataModelCoords.leadtime.name: [np.timedelta64(1, "h"), np.timedelta64(2, "h")],
            extradim: [1, 2],
        },
    )

    output = DataModel._create_intermediate_dataset(
        inputdataset,
        inputdataset.coords,
        np.timedelta64(1, "h"),
    )

    # The time dimension of the output is expected to only have 3 values, the first value from the
    #  input time should be gone
    assert np.array_equal(
        output[DataModelCoords.time.name].data,
        pd.date_range(
            np.datetime64("2014-09-06") + np.timedelta64(1, "h"),
            periods=4 - 1,
            freq="h",
        ).values,
    )
    # Check the dims of the output variables are in the expected order
    #   The order also determines how the values should be compared
    assert output["obsvar1"].dims == ("time",)
    assert output["simvar1"].dims == ("time", "leadtime")
    assert output["obsvar2"].dims == ("extradim", "time")
    assert output["simvar2"].dims == ("extradim", "time", "leadtime")
    # Check the values of the output variables
    assert np.array_equal(
        output["obsvar1"].data,
        [2.0, 3.0, 4.0],
        equal_nan=True,
    )
    assert np.array_equal(
        output["simvar1"].data,
        [[4.0, np.nan], [14.0, 6.0], [np.nan, 16.0]],
        equal_nan=True,
    )
    assert np.array_equal(
        output["obsvar2"].data,
        [
            [6.0, 9.0, 12.0],
            [8.0, 12.0, 16.0],
        ],
        equal_nan=True,
    )
    assert np.array_equal(
        output["simvar2"].data,
        [
            [[10.0, np.nan], [42.0, 15.0], [np.nan, 48.0]],
            [[14.0, np.nan], [56.0, 21.0], [np.nan, 64.0]],
        ],
        equal_nan=True,
    )
