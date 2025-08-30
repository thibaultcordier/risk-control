# Risk Control Project (mlrisko)

[![CI/CD Pipeline](https://github.com/thibaultcordier/risk-control/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/thibaultcordier/risk-control/actions)
[![Code Coverage](https://codecov.io/gh/thibaultcordier/risk-control/branch/main/graph/badge.svg)](https://codecov.io/gh/thibaultcordier/risk-control)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://badge.fury.io/py/mlrisko.svg)](https://badge.fury.io/py/mlrisko)
[![Documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?colorB=319795)](https://thibaultcordier.github.io/risk-control/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-red.svg)](https://github.com/astral-sh/ruff)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](https://mypy-lang.org/)

**mlrisko** (MLRiskControl) is a comprehensive toolkit for implementing risk control mechanisms for predictive algorithms based on the paper "Learn then test: Calibrating predictive algorithms to achieve risk control" by Angelopoulos et al. (2025).

The primary goal is to ensure that machine learning algorithms perform reliably and maintain a controlled level of risk through advanced calibration techniques.

## Installation

To install the necessary dependencies, run:

```bash
uv sync
uv pip install -e .
```

For development purposes, you can install the development dependencies with:
```bash
uv sync --all-groups
```

## Running the Example

To run the example, execute the following command:

```bash
uv run python examples/plot_regression.py
uv run python examples/plot_classification.py
uv run python examples/plot_classification_bis.py
```

## Documentation

For detailed documentation, refer to the [docs](https://thibaultcordier.github.io/risk-control/).

Or you can build the documentation with:
```bash
uv run mkdocs serve
```

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## References

Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2025). Learn then test: Calibrating predictive algorithms to achieve risk control. The Annals of Applied Statistics, 19(2), 1641-1662.
