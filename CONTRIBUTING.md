# Contributing

Currently **wm_planners** is closed/private for collaborators only. 

## Workflow
### Action Item
* Issue/sub-issue to be created as part of a timeline block,

### Issue on Linear
* Create the issue on LinearApp (description, priority, assign, label, milestone, and status)

### Development
* Create MVP for a particular solution
* Evaluate on test cases

### Draft a Pull Request
* Summarize contributions and results,
* Lint and fix any bugs,
* Documentation

### Review Process
* Assign reviewer


![Workflow](docs/assets/workflow.gif)

---

## Quick steps to contribute:
1. Obtain a branch name for the issue you're working on in Linear.
![Branch copy](docs/assets/branch_copy.gif)
2. Create a feature branch from `devel` with name from step 1 (e.g., implement-model-free-baseline-ahx-78)
3. Install development dependencies: `pip install -e .[dev,docs]`.
4. Run quality checks: `ruff check`, `pylint`, `pytest`, `mypy`.
5. Open a pull request with clear scope and test evidence.
