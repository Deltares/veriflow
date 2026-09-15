=========================
Datamodel
=========================

.. autosummary::
   :toctree: _generated

   veriflow.datatree.datatree

VeriflowDataTree
================

``veriflow.datatree.datatree.VeriflowDataTree`` is a typing-only helper, not a real runtime
class. It exists so that static type checkers (mypy, pyright) know about the ``veriflow``
accessor added to ``xr.DataTree`` by :class:`~veriflow.datatree.datatree.VeriflowAccessor`,
without veriflow having to subclass or monkeypatch xarray's actual ``DataTree`` type.

At runtime, ``VeriflowDataTree`` is simply an alias for ``xr.DataTree`` (there is no separate
class, and no ``isinstance`` distinction). When you receive or construct a plain ``xr.DataTree``
and want typed access to ``.veriflow``, cast it once:

.. code-block:: python

    from typing import cast
    from veriflow.datatree.datatree import VeriflowDataTree

    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow-datatree"))
    dt.veriflow.add_input_data(...)  # now statically typed

See :class:`~veriflow.datatree.datatree.VeriflowAccessor` for the actual runtime API surface
exposed through ``.veriflow``.
