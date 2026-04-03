[![Tests](https://github.com/Deltares/scoreflow/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/scoreflow/actions/workflows/ci.yml)
[![Docs](https://github.com/Deltares/scoreflow/actions/workflows/docs.yml/badge.svg)](https://github.com/Deltares/scoreflow/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/Deltares/scoreflow//branch/main/graph/badge.svg)](https://codecov.io/gh/Deltares-research/DPyVerification)

# Scoreflow


A verification pipeline, that integrates the full verification process
- 📥 Fetching data
- 🧮 Computing scores
- 📝 Writing results

## Key features
- ✅ Full control over the verification pipeline via configuration
- ✅ Native integration with [Delft-FEWS](https://oss.deltares.nl/web/delft-fews) 
- ✅ Builds on [Scores](https://scores.readthedocs.io/en/stable/) for computation of scores. This package has extensive functionality, and it's documentation is world-class.
- ✅ Extensible with your own (private) datasources, scores and datasinks
- ✅ Optimized internal datamodel for efficient computation


## 👥 Who Is This For?

This project is aimed at anyone who's interested in assessing model and forecast quality in an easy and reproducible way, like:
- operational forecasters
- model developers
- researchers and data-scientists

This package is developed by people working in the real-time hydro- and meteorological forecasting domain, but the software is not limited to this scope.
## Why this package?

Verification is an essential part of model development and forecasting. However, setting up a verification pipeline for any given use-case, may be time-consuming and technically complex. This package aims to lower the bar for verification, by saving users time and handling some of the technical challenges in verification. We aim to make verification process easier, by handling all steps of the pipeline with just one easy to read user-configuration. In addition to making verification more accessible, we hope to make the process more robust, transparent and fully reproducible so you can focus on what the results actually tell you. 

Some of the technical challenges of verification include:

- Computation of metrics can be complex
- Data volumes can become larger than memory
- Data transformations may be required before computation
- Data may have to be ingested from various sources (like files or databases)
- Data may have to be written to various data destinations (like files or databases)


Naturally, verification is nothing new and many custom approaches exist. Custom approaches may aim to handle each of the necessary steps in the verification pipeline for a given use-case. Some of the possible advantages of this software, compared to custom approaches include:
- This software is versioned, tested and published, which improves reliability
- This software is documented which offers transparency to users
- This software runs on well documented configuration, ensuring ease of use when making instructions for a pipeline.
- Any verification pipeline is fully-transferable to other users or systems, because it only relies on a single config file.
- The pipeline returns standardized output, which can be directly inspected in any environment. When working in an interactive Python environment, results can be directly returned from the pipeline.
- This software is flexible. As a developer, you can write your own plug-ins for datasources, scores and datasinks. In this way, you can tailor this framework to your own use-case.


## Technical features
- Builds on [Xarray](https://docs.xarray.dev/en/stable/#) for handling multidimensional data. 
- Supports [Zarr](https://zarr.dev/) for cloud-friendly data storage
- Supports [Dask](https://www.dask.org/) for parallel and lazy computation

## Installation

Install from source:

```bash
git clone https://github.com/Deltares-research/DPyVerification.git
cd DPyVerification
pip install .
```

As a developer: see CONTRIBUTING.md
