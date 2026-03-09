# DPyVerification


A verification pipeline, that integrates the full verification process
- 📥 Fetching observations and simulations
- 🧮 Compting metrics
- 📝 Writing results

## Key features
- ✅ Full control over the verification pipeline via configuration
- ✅ Native integration with [Delft-FEWS](https://oss.deltares.nl/web/delft-fews) 
- ✅ Builds on [Scores](https://scores.readthedocs.io/en/stable/) for computation of scores
- ✅ Extendable with your own (private) datasources, scores and datasinks
- ✅ Optimized internal datamodel for efficient computation

## Technical features
- ✅ Builds on [Xarray](https://docs.xarray.dev/en/stable/#) for handling multidimensional data. 
- ✅ Supports [Zarr](https://zarr.dev/) for cloud-friendly data storage
- ✅ Supports [Dask](https://www.dask.org/) for parallel and lazy computation


## 🤔 Why this package?

This software aims to make the verification process easy, reproducible and reliable, so you can focus on what the results actually tell you.

Easy
- Custom verification methods are time-consuming and complex* to set-up.
- Custom verification methods may lack flexibility to handle other use-cases or configurations

Reproducible
- Custom methods may lead to different results
- Custom methods may be hard to reproduce

Reliable
- Custom verification methods may be untested, unversioned or unmaintained

**Fetching data from various sources, transforming it into the right datamodel, computing, writing and visualizing results...*
## 👥 Who Is This For?

This project is aimed at people working in hydrometeorological forecasting and modelling, including:
- operational forecasting centers
- researchers and data-scientists evaluating models and forecast systems


If you work with rainfall, discharge, floods, tides, or related models, forecasts and data. 

## Installation

From pypi: not available yet

As a developer: see CONTRIBUTING.md
