# DS_GA_3001
This repo supports the DS_GA_3001 - Embodied Learning and Vision Course Project

## Features

- **Modern packaging** – PEP 517/518/621 ``pyproject.toml`` using Hatchling.
- **Editable installs** – ``pip install -e .[dev,docs]`` for iterative notebook
  development.
- **Documentation-first** – Sphinx configured with MyST-NB and a notebooks
  gallery ready for Read the Docs or local builds.
- **Testing** – Pytest suite to validate core functionality.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,docs]
```

Run the example notebook by launching Jupyter Lab and opening
``docs/notebooks/getting_started.md``.  Build the documentation locally with:

```bash
sphinx-build -b html docs docs/_build/html
```

Execute the tests:

```bash
pytest
```

## Project Solution
| Authors              | Email                |
|-------------------|----------------------|
| **Long Quang**    | lq2146@nyu.edu       |
| **Akash Guna** | ar9799@nyu.edu     |
| **Deepanshu Mody** | dm6262@nyu.edu      |

<sup>
Authors are with New York University.
</sup>

[![Tests](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/tests.yml/badge.svg)](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/tests.yml)
![Pylint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lquan9/d09cfb9f249c9995b9fa1943d90a1d71/raw/ds3001-pylint.json)
[![Docs](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/docs.yml/badge.svg)](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/docs.yml)
[![Python Version](https://img.shields.io/badge/python-%203.10-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Code](#) | [Paper](#)

## License

This project is licensed under the terms of the MIT license.  See
[LICENSE](LICENSE).

