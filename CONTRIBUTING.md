# 🤝 Contributing to the project 
We warmly welcome contributions! The guideline below aims to support you in the process.

Currently, we have guidelines for the following types of contributions:

[Code contributions](#code-contributions)


## Code contributions

veriflow requires Python 3.11 or newer.

### GitHub
 If you are new to Git and pull request based development, GitHub provides a
[guide](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) you will find helpful.

### Fork the project
1. Create your own fork of the project using the GitHub web user interface.
2. Clone your fork. Avoid cloning (https://github.com/Deltares/veriflow).
3. Immediately create a new local branch, with a command such as git checkout -b branch_name.

### Create a GitHub issue
Prior to developing a pull request, considering creating a GitHub issue to capture what the pull request is trying to achieve. Otherwise, please explain this in the pull request.

### Set up for local development

veriflow is developed using uv. Refer to the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) to install uv.

#### Create a virtual environment and install the project.

`uv sync --dev`

#### Activate the virtual environment
- Linux / macOS -> `source .venv/bin/activate`
- Windows ->  `.venv/Scripts/activate`


#### Make your changes
You're ready to make changes. Please try to write clear code, use short and functional names for functions and classes and document using docstrings. Where needed, provide line comments for clarity.

#### Run the development tools
We use pytest for testing, ruff for linting and code formatting and mypy for type checking.

- `uv run pytest` 
- `uv run ruff check` 
- `uv run mypy`

Ruff formatting and linting is also part of the pre-commit hook.


#### Commit your changes to your branch
Please use separate commits for separate topics or actions. Use the commit message to concisely describe what the commit is about. Always start with a capital letter and use a reference to an issue number when applicable.

#### Push your branch 
Push your work to the remote. Please make sure all tests are green.

#### Request a review by assigning a reviewer
Each PR will be reviewed by the development team, before merging to main. Please request a review by assigning the PR to a member of the team. 
