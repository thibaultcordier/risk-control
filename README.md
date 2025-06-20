# Risk Control Project

This project focuses on developing and implementing risk control mechanisms for predictive algorithms based on the paper "Learn then test: Calibrating predictive algorithms to achieve risk control" by Angelopoulos et al. (2025).
The primary goal is to ensure that the algorithms perform reliably and maintain a controlled level of risk.

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

For detailed documentation, refer to the [docs](docs/index.md).

Or you can build the documentation with:
```bash
uv run mkdocs serve
```

## License

This project is licensed under the MIT License.

## References

Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I., & Lei, L. (2025). Learn then test: Calibrating predictive algorithms to achieve risk control. The Annals of Applied Statistics, 19(2), 1641-1662.
