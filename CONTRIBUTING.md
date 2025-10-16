# Contributing to DPyVerification

## How to contribute

### Code contributions

#### Local development

DPyVerification's development toolchain requires Python 3.11 or newer.

DPyVerification is developed using Poetry. Refer to the [Poetry documentation](https://python-poetry.org) to install Poetry.

You should first fork the DPyVerification repository and then clone it locally, so that you can make pull requests against the
project. If you are new to Git and pull request based development, GitHub provides a
[guide](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) you will find helpful.

Next, you should tell Poetry to use a correct python installation (only if python 3.11+ is not the default python), install DPyVerification's dependencies, and run the test suite to make sure everything is working as expected. Within the main directory of the local checkout of DPyVerification:

```bash
poetry env use PATH_TO_THE_PYTHON_EXECUTABLE
poetry install
poetry run pytest
pre-commit install # Optional
```

When you contribute to DPyVerification, automated tools will be run to make sure your code is suitable to be merged. All of these are included in the pytest checks. However, it may be useful during development to run them separately to automatically fix problems. And to integrate them in your development environment.

Tools that are included are ruff for formatting, ruff for linting, mypy for type checking.

TODO: info on how to run these manually, including autofix options
TODO: actually include these in the pytest suite, so no need for pre-commit for now.
