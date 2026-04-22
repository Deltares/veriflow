Configuration
==================

The pipeline configuration can be provided in either Python objects, or via a YAML configuration file. For new users, we recommend starting with a YAML configuration file. 

.. tab-set::

   .. tab-item:: YAML

      .. code-block:: yaml

         verification:
           enabled: true
           mode: strict

   .. tab-item:: Python

      .. code-block:: python

         from veriflow import configure

         configure(
             verification={
                 "enabled": True,
                 "mode": "strict",
             }
         )

