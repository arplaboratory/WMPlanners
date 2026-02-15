# Contributing

Currently **DS_GA_3001** is closed/private for collaborators only. We use [LinearApp](https://linear.app/ahxrn/project/embodied-learning-7b745e425744/overview) to track issues.
Steps to contribute:
1. Obtain a branch name for the issue you're working on in Linear. 
2. Create a feature branch from `devel` with name from step 1 (e.g., implement-model-free-baseline-ahx-78)
![Branch copy](docs/assets/branch_copy.gif)
3. Install development dependencies: `pip install -e .[dev,docs]`.
3. Run quality checks: `ruff check`, `pylint`, `pytest`, `mypy`.
4. Open a pull request with clear scope and test evidence.

