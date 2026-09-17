"""The veriflow data tree structure and accessor.

The veriflow DataTree is an instance of ``xarray.DataTree`` and thus has all standard methods and
attributes of an ``xarray.DataTree``. The VeriflowAccessor extends this functionality with methods
specific to the veriflow DataTree structure. It can be used as follows:

.. code-block:: python

    import xarray as xr

    from veriflow import run_pipeline

    # The output of the veriflow pipeline is an instance of ``xr.DataTree``.
    dt: xr.DataTree = run_pipeline(...)

    # It has all the standard methods and attributes of an ``xarray.DataTree``.
    dt.keys()  # Example of a standard xarray.DataTree method.

    # The ``dt.veriflow`` accessor provides additional methods to interact with the veriflow
    # DataTree.

    # Example 1: Get the list of verification pair IDs.
    dt.veriflow.verification_pairs

    # Example 2: List scores for a given verification_pair
    dt.veriflow.list_scores("example_verification_pair_id")

The VeriflowAccessor extends ``xarray.DataTree`` via the ``xr.register_datatree_accessor``
decorator, as recommended by the `xarray documentation
<https://docs.xarray.dev/en/latest/internals/extending-xarray.html>`_.
"""

from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import xarray as xr

from veriflow.configuration.utils import VerificationPair
from veriflow.constants import (
    FORECAST_DATA_TYPES,
    HISTORICAL_DATA_TYPES,
    DataType,
    SpatialType,
    StandardAttribute,
    StandardDim,
)

__all__ = ["VeriflowAccessor"]


@xr.register_dataset_accessor("verification")  # type:ignore[no-untyped-call, misc]
class InputDatasetExtension:
    """xr.Dataset representing a specific data type from a specific source.

    xr.register_dataset_accessor is the recommended way to extend xr.Dataset.
    see: https://docs.xarray.dev/en/stable/internals/extending-xarray.html. It's used here to
    extend the input datasets so we can directly access properties (like data type and
    source) and the validation method that checks the input dataset against a schema.

    The dataset is expected to carry ``data_type`` and ``source`` keys on its ``attrs``.
    Each data variable in the dataset represents a physical variable, and is expected to
    carry a ``units`` key on its ``attrs``.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj
        self.validate_attr_exists("data_type")
        self.validate_attr_exists("source_id")

    def validate_attr_exists(self, attr_name: str) -> None:
        """Validate that the specified attribute exists in the dataset."""
        if attr_name not in self._obj.attrs:  # type:ignore[misc]
            msg = f"No {attr_name} set on {self._obj} attrs."
            raise ValueError(msg)

    @property
    def data_type(self) -> DataType:
        """The data type of the dataset."""
        self.validate_attr_exists("data_type")
        return DataType(self._obj.attrs["data_type"])  # type:ignore[misc]

    @property
    def spatial_type(self) -> SpatialType:
        """The spatial type of the dataset (defaults to ``point`` when unset)."""
        return SpatialType(self._obj.attrs.get("spatial_type", SpatialType.point))  # type:ignore[misc]

    @property
    def is_thresholds(self) -> bool:
        """Boolean indicating this dataset is a thresholds dataset."""
        return self.data_type == DataType.threshold

    @property
    def is_historical(self) -> bool:
        """Boolean indicating this dataset is historical."""
        return self.data_type in HISTORICAL_DATA_TYPES

    @property
    def is_forecast(self) -> bool:
        """Boolean indicating this dataset is a forecast."""
        return self.data_type in FORECAST_DATA_TYPES

    @property
    def source_id(self) -> str:
        """The source ID."""
        self.validate_attr_exists("source_id")
        return str(self._obj.attrs["source_id"])  # type:ignore[misc]


class DataTreeNode(StrEnum):
    """Enum for standard DataTree node names."""

    ROOT = "veriflow-datatree"
    INPUT_DATA = "input_data"
    ALIGNED_INPUT = "aligned_input"
    OUTPUT = "output"
    OBSERVATIONS = "observations"
    SIMULATIONS = "simulations"


@xr.register_datatree_accessor("veriflow")  # type: ignore[no-untyped-call, misc]
class VeriflowAccessor:
    """Accessor for managing the internal datatree for veriflow.

    The DataTree layout is as follows:

    - The root node is ``veriflow-datatree``.
    - The the root node's child nodes are ``input_data`` and one or more verification pairs.
    - The ``input_data`` node holds the validated input datasets. These are instances of
      `xr.Dataset`as fetched from the configured datasources. They are instances of `xr.Dataset`
      keyed by their ``source_id``.
    - Each ``verification_pair``  node represents a verification pair, identified by its unique ID,
      and has two main child nodes: ``aligned_input`` and ``output``.

      - The ``aligned_input`` node contains data prepared for verification. The
        observation and simulation data are aligned and ready for verification. For
        example, when a forecast is verified against observations, the observations
        are mapped into forecast space along the ``forecast_reference_time`` and
        ``lead_time`` dimensions.

      - The ``output`` node contains the results of the verification. Each child
        under the ``output`` node is a dataset containing one or more data
        variables corresponding to different aspects of the output.

    An example of the DataTree structure::

        veriflow-datatree
        ├── input_data
        │   ├── source_id_a
        │   └── source_id_b
        │   └── source_id_c
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
        return [name for name in self.dt.children if name != DataTreeNode.INPUT_DATA]

    @staticmethod
    def map_historical_into_forecast_space(
        obs: xr.DataArray,
        sim: xr.DataArray,
    ) -> xr.DataArray:
        """
        Transform array of historical data into forecast structure.

        Given an observation array with dimension 'time' and a simulation array with
        dimensions 'forecast_reference_time' and 'lead_time', project the observed
        values onto the simulation array.

        This method is called at runtime when the pipeline starts a score computation on forecast
        data. On the fly, the observation array is mapped to the forecast structure, so data are
        aligned along the same dimensions.
        """
        # Stack forecast time axes
        stacked_time = sim[StandardDim.time].stack(
            z=(StandardDim.forecast_reference_time, StandardDim.lead_time),
        )

        # Reindex observations onto stacked forecast times
        obs_aligned = obs.reindex(
            time=stacked_time.to_numpy(),  # type:ignore[misc]
        )

        # Attach forecast coordinates explicitly (from the MultiIndex)
        z_index = stacked_time.indexes["z"]  # type:ignore[misc]

        # Assign forecast_reference_time and lead_time coordinates to the aligned
        # observations, based on the MultiIndex of the stacked time dimension. This is
        # necessary because after re-indexing, the original time dimension of the observations
        # is now aligned with the stacked time dimension of the simulations, which has a MultiIndex
        # of forecast_reference_time and lead_time.
        obs_aligned = obs_aligned.assign_coords(
            forecast_reference_time=(  # type:ignore[misc]
                StandardDim.time,
                z_index.get_level_values(StandardDim.forecast_reference_time),  # type:ignore[misc]
            ),
            lead_time=(  # type:ignore[misc]
                StandardDim.time,
                z_index.get_level_values(StandardDim.lead_time),  # type:ignore[misc]
            ),
        )

        # Set the time coordinate to be the stacked time (MultiIndex of forecast_reference_time and
        # lead_time)
        obs_indexed = obs_aligned.set_index(
            time=(StandardDim.forecast_reference_time, StandardDim.lead_time),
        )

        # Unstack into forecast space
        obs_projected = obs_indexed.unstack(StandardDim.time)

        # Preserve attrs
        obs_projected.attrs = obs.attrs

        return obs_projected

    def get_pair(
        self,
        verification_pair: VerificationPair,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Return observations and simulations for a given verification pair.

        Selects ``verification_pair.variable`` from each source's dataset and returns the
        resulting DataArrays. This method is called by the verification pipeline at runtime
        to retrieve the correct data for one of the configured verification pairs.
        """
        obs_ds = self._get_input_data(verification_pair.observations_source_id)
        sim_ds = self._get_input_data(verification_pair.simulations_source_id)

        variable = verification_pair.variable
        if variable not in obs_ds.data_vars:
            msg = (
                f"Variable '{variable}' configured on verification pair "
                f"'{verification_pair.id}' not found in observations source "
                f"'{verification_pair.observations_source_id}'. "
                f"Available variables: {sorted(obs_ds.data_vars)}."  # type:ignore[type-var]
            )
            raise ValueError(msg)
        if variable not in sim_ds.data_vars:
            msg = (
                f"Variable '{variable}' configured on verification pair "
                f"'{verification_pair.id}' not found in simulations source "
                f"'{verification_pair.simulations_source_id}'. Available variables: "
                f"{sorted(sim_ds.data_vars)}."  # type:ignore[type-var]
            )
            raise ValueError(msg)

        obs = obs_ds[variable]
        sim = sim_ds[variable]

        def _propagate_attrs(da: xr.DataArray, ds: xr.Dataset) -> None:
            """Propagate dataset-level attrs needed downstream onto the extracted DataArray."""
            # data_type, so downstream code (scores etc.) can read it via the data array's attrs.
            da.attrs.setdefault("data_type", ds.attrs.get("data_type"))  # type:ignore[misc]
            # spatial_type, defaulting to point so the full data type is known.
            da.attrs.setdefault(  # type:ignore[misc]
                "spatial_type",
                ds.attrs.get("spatial_type", SpatialType.point),  # type:ignore[misc]
            )
            # CRS, so downstream reprojection knows the source CRS. Every validated dataset is
            # guaranteed to carry a ``crs`` attribute (defaulting to EPSG:4326).
            da.attrs.setdefault(  # type:ignore[misc]
                StandardAttribute.crs,
                ds.attrs[StandardAttribute.crs],  # type:ignore[misc]
            )
            # source_id
            da.attrs.setdefault(  # type:ignore[misc]
                StandardAttribute.source_id,
                ds.attrs.get(StandardAttribute.source_id),  # type:ignore[misc]
            )

        _propagate_attrs(obs, obs_ds)
        _propagate_attrs(sim, sim_ds)

        if sim_ds.verification.is_forecast:  # type:ignore[misc]
            # Map historical into forecast space upon score computation
            return self.map_historical_into_forecast_space(obs, sim), sim

        # If the simulation is not a forecast, it is a historical data type (an observation or
        #   historical simulation). In this case: verify along dimension 'time' instead of mapping
        #   data into forecast space.
        return obs, sim

    def get_thresholds_array(self, variable: str) -> xr.DataArray:
        """Get the thresholds array for a given variable from the input dataset."""
        if self._path_exists_in_dt(DataTreeNode.INPUT_DATA):
            input_data_node = cast("xr.DataTree", self.dt[DataTreeNode.INPUT_DATA])
            for child in input_data_node.children.values():
                dataset = cast("xr.Dataset", child.to_dataset())
                if not dataset.verification.is_thresholds:  # type:ignore[misc]
                    continue
                if variable not in dataset.data_vars:
                    msg = (
                        f"Variable '{variable}' not found in thresholds dataset. "
                        f"Available variables: {sorted(dataset.data_vars)}."  # type:ignore[type-var]
                    )
                    raise ValueError(msg)
                return dataset[variable]
        msg = (
            "No thresholds dataset found in the input dataset, but required for computing "
            "categorical scores."
        )
        raise ValueError(msg)

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

        # Add observations and simulations data. We use `to_dataset()` to preserve the variable
        # names and ensure consistency in the DataTree structure.
        self.dt[f"{base_path_in_dt}/{DataTreeNode.OBSERVATIONS}"] = obs.to_dataset()
        self.dt[f"{base_path_in_dt}/{DataTreeNode.SIMULATIONS}"] = sim.to_dataset()

    def add_input_data(self, data: Iterable[xr.Dataset]) -> None:
        """Add raw input datasets to the datatree, keyed by their source_id."""
        for dataset in data:
            path_in_dt = f"{DataTreeNode.INPUT_DATA}/{dataset.verification.source_id}"  # type:ignore[misc]
            self._validate_path_does_not_exist(path_in_dt)
            self.dt[path_in_dt] = dataset

    def _get_input_data(self, source_id: str) -> xr.Dataset:
        """Get a raw input dataset previously added via ``add_input_data``, by source_id."""
        return cast(
            "xr.Dataset",
            self.dt[f"{DataTreeNode.INPUT_DATA}/{source_id}"].to_dataset(),
        )

    def filter_nodes(
        self,
        *,
        include_input_data: bool,
        include_aligned_input_data: bool,
        include_output: bool,
    ) -> xr.DataTree:
        """Filter the datatree based on the specified inclusion flags.

        Builds a new DataTree containing only the parts of this DataTree selected by
        ``config``: the raw ``input_data`` node, and/or the ``aligned_input`` and ``output``
        nodes under each verification pair, in any combination.
        """
        filtered_dt = cast("xr.DataTree", xr.DataTree(name=self.dt.name))

        if include_input_data and self._path_exists_in_dt(DataTreeNode.INPUT_DATA):
            filtered_dt[DataTreeNode.INPUT_DATA] = self.dt[DataTreeNode.INPUT_DATA]

        for pair_id in self.verification_pairs:
            if include_aligned_input_data:
                path = f"{pair_id}/{DataTreeNode.ALIGNED_INPUT}"
                if self._path_exists_in_dt(path):
                    filtered_dt[path] = self.dt[path]
            if include_output:
                path = f"{pair_id}/{DataTreeNode.OUTPUT}"
                if self._path_exists_in_dt(path):
                    filtered_dt[path] = self.dt[path]

        return filtered_dt


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
