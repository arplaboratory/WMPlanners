# Testing wm_planner

Run tests from the repository root after installing the project and its `dev` dependencies:

```bash
pytest
```

The default run includes a small integration check against the public Hugging Face dataset. It fetches four rows and their images through the public dataset API, then checks that `HFWindowDataset` can produce a training window.

After generating a local dataset with the pipeline, you can also check its saved Hugging Face output:

```bash
pytest --pipeline-out pipeline_out
```

To run one module during development, use `pytest tests/test_hfwindowdataset.py`.

## Writing tests here

- Put a test next to related tests in the corresponding `test_*.py` module. Name it for the behavior it checks.
- Prefer a small, deterministic fixture. Use a seeded NumPy generator, `tmp_path` for files, and `monkeypatch` or `unittest.mock` for external writes such as Hub pushes.
- Assert observable outputs and invariants. Avoid assertions about values calculated entirely inside the test or tests that print output without asserting anything.
- Run the affected module while developing, then run `pytest` before merging.
