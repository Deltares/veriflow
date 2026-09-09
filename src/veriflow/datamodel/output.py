"""The veriflow data tree structure and accessor.

The veriflow DataTree is an instance of ``xarray.DataTree`` and thus has all standard methods and
attributes of an ``xarray.DataTree``. The VeriflowAccessor extends this functionality with methods
specific to the veriflow DataTree structure. It can be used as follows:

.. code-block:: python

    import xarray as xr

    from veriflow import run_pipeline

    # The output of the veriflow pipeline is an instance of ``xr.DataTree``.
    dt: xr.DataTree = run_pipeline(...)

    # The ``dt.veriflow`` accessor provides convenient methods to interact with the veriflow
    # DataTree. Get the list of verification pair IDs.
    dt.veriflow.verification_pairs

The VeriflowAccessor extends ``xarray.DataTree`` via the ``xr.register_datatree_accessor``
decorator, as recommended by the `xarray documentation
<https://docs.xarray.dev/en/latest/internals/extending-xarray.html>`_.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, cast

import xarray as xr

from veriflow.configuration.utils import VerificationPair

__all__ = ["VeriflowAccessor"]


class DataTreeNode(StrEnum):
    """Enum for standard DataTree node names."""

    ROOT = "veriflow-output"
    OUTPUT = "output"
    ALIGNED_INPUT = "aligned_input"
    OBSERVATIONS = "observations"
    SIMULATIONS = "simulations"


@xr.register_datatree_accessor("veriflow")  # type: ignore[no-untyped-call, misc]
class VeriflowAccessor:
    """Accessor for managing the output dataset for veriflow.

    The DataTree layout is as follows:

    - The root node is ``veriflow-output``.
    - Each child node represents a verification pair, identified by its unique ID.
    - Each verification pair has two main child nodes: ``aligned_input`` and ``output``.

      - The ``aligned_input`` node contains data prepared for verification. The
        observation and simulation data are aligned and ready for verification. For
        example, when a forecast is verified against observations, the observations
        are mapped into forecast space along the ``forecast_reference_time`` and
        ``lead_time`` dimensions.

      - The ``output`` node contains the results of the verification. Each child
        under the ``output`` node is a dataset containing one or more data
        variables corresponding to different aspects of the output.

    Schematic representation of the DataTree structure::

        veriflow-output
        ├── verification_pair_1
        │   ├── aligned_input
        │   │   ├── observations
        │   │   │   └── waterlevel
        │   │   └── simulations
        │   │       └── waterlevel
        │   └── output
        │       ├── crps
        │       │   ├── crps
        │       │   ├── overforecast_penalty
        │       │   ├── underforecast_penalty
        │       │   └── forecast_spread_term
        │       └── rank_histogram
        │           └── rank_histogram
        └── verification_pair_2
            ├── aligned_input
            │   └── ...
            └── output
                └── ...
    """

    def __init__(self, dt: xr.DataTree) -> None:
        self._dt = dt

    @property
    def dt(self) -> xr.DataTree:
        """Get the underlying DataTree."""
        return self._dt

    @property
    def verification_pairs(self) -> list[str]:
        """Get a list of verification pair IDs in the DataTree."""
        return list(self.dt.children.keys())

    def get_verification_pair_node(self, verification_pair_id: str) -> xr.DataTree:
        """Get the DataTree node corresponding to a specific verification pair."""
        return cast("xr.DataTree", self.dt[verification_pair_id])

    def get_aligned_input(self, verification_pair_id: str) -> xr.DataTree:
        """Get the input dataset for a specific verification pair."""
        pair_node = self.get_verification_pair_node(verification_pair_id)
        return cast("xr.DataTree", pair_node[DataTreeNode.ALIGNED_INPUT])

    def get_outputs(self, verification_pair_id: str) -> xr.DataTree:
        """Get the output dataset for a specific verification pair."""
        pair_node = self.get_verification_pair_node(verification_pair_id)
        return cast("xr.DataTree", pair_node[DataTreeNode.OUTPUT])

    def list_scores(self, verification_pair_id: str) -> list[str]:
        """Get a list of available scores for a specific verification pair."""
        output_node = self.get_verification_pair_node(verification_pair_id)[DataTreeNode.OUTPUT]
        return list(output_node.children.keys())  # type:ignore[misc]

    def _path_exists_in_dt(self, path_in_dt: str) -> bool:
        """Validate if a given path does not exist in the DataTree."""
        try:
            self.dt[path_in_dt]
        except KeyError:
            return False
        else:
            return True

    def _validate_path_does_not_exist(self, path_in_dt: str) -> None:
        """Validate the path does not exist in the DataTree. Raise ValueError if it does."""
        if self._path_exists_in_dt(path_in_dt):
            msg = f"Path '{path_in_dt}' already exists in the DataTree."
            raise ValueError(msg)

    def add_score(
        self,
        verification_pair: VerificationPair,
        result: xr.DataArray | xr.Dataset,
        name: str,
    ) -> None:
        """Add a score results to the datatree."""
        # We always convert the result to a Dataset for consistency, even if it's a DataArray.
        # This results in a clean tree structure that is easier to understand and navigate.
        if isinstance(result, xr.DataArray):  # type:ignore[misc]
            result = result.to_dataset()

        path_in_dt = f"{verification_pair.id}/{DataTreeNode.OUTPUT}/{name}"
        self._validate_path_does_not_exist(path_in_dt)
        self.dt[path_in_dt] = result

    def add_staged_input_data(
        self,
        verification_pair: VerificationPair,
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> None:
        """Add input data to the datastore."""
        base_path_in_dt = f"{verification_pair.id}/{DataTreeNode.ALIGNED_INPUT}"
        self._validate_path_does_not_exist(base_path_in_dt)

        # Add reference data. We use `to_dataset()` to preserve the variable names and ensure
        # consistency in the DataTree structure.
        self.dt[f"{base_path_in_dt}/{DataTreeNode.OBSERVATIONS}"] = obs.to_dataset()
        self.dt[f"{base_path_in_dt}/{DataTreeNode.SIMULATIONS}"] = sim.to_dataset()


if TYPE_CHECKING:

    class VeriflowDataTree(xr.DataTree):  # type: ignore[no-untyped-call]
        """A DataTree with the runtime-registered `veriflow` accessor, for static typing.

        `xr.DataTree` instances are never actually instances of this class at runtime; use
        `cast("VeriflowDataTree", dt)` once after creating/receiving a DataTree to get typed
        access to `dt.veriflow` for the rest of that scope.
        """

        veriflow: VeriflowAccessor
else:
    VeriflowDataTree = xr.DataTree
