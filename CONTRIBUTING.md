# Contributing to Risk Control

Thank you for your interest in contributing to the Risk Control project! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Code Review Process](#code-review-process)

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/risk-control.git
   cd risk-control
   ```
3. **Set up the development environment** (see below)

## Development Setup

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Install dependencies**:
   ```bash
   uv sync --all-groups
   ```

2. **Install the package in development mode**:
   ```bash
   uv pip install -e .
   ```

3. **Set up pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

## Making Changes

### Branch Strategy

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the guidelines below

3. **Keep commits atomic** and well-described

### Commit Message Format

Use conventional commit messages:
```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(decision): add new classification decision algorithm`
- `fix(parameter): resolve issue with parameter estimation`
- `docs(readme): update installation instructions`

## Code Style

We use several tools to maintain code quality:

### Ruff (Linting & Formatting)
```bash
# Check code style
uv run make lint

# Format code
uv run make format
```

### MyPy (Type Checking)
```bash
# Run type checking
uv run make type-check
```

### Pre-commit Hooks
Pre-commit hooks run automatically on commit and include:
- Code formatting with Ruff
- Import sorting
- Basic linting checks

## Testing

### Running Tests
```bash
# Run all tests
uv run make tests

# Run tests with coverage
uv run make report-tests
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Aim for high test coverage
- Test both success and failure cases

### Test Structure
```python
def test_feature_name():
    """Test description."""
    # Arrange
    # Act
    # Assert
```

## Documentation

### Code Documentation
- Use docstrings for all public functions and classes
- Follow NumPy docstring format
- Include type hints for all function parameters

### Building Documentation
```bash
# Build documentation
uv run make docs

# Serve documentation locally
uv run make serve
```

### Documentation Guidelines
- Keep documentation up to date with code changes
- Include examples for complex functionality
- Update README.md for user-facing changes

## Submitting Changes

### Pull Request Process

1. **Ensure your code passes all checks**:
   ```bash
   uv run make lint
   uv run make type-check
   uv run make tests
   ```

2. **Update documentation** if needed

3. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference to any related issues
   - Screenshots for UI changes (if applicable)

4. **Wait for CI checks** to pass

### Pull Request Template

We use a pull request template to ensure all necessary information is included. The template will be automatically loaded when you create a new pull request. You can find the template at [.github/pull_request_template.md](.github/pull_request_template.md).

## Code Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **At least one maintainer** must approve the PR
3. **Address review comments** promptly
4. **Maintainers will merge** when ready

### Review Guidelines

**For Contributors:**
- Respond to review comments promptly
- Be open to feedback and suggestions
- Ask questions if something is unclear

**For Reviewers:**
- Be constructive and respectful
- Focus on the code, not the person
- Provide specific, actionable feedback

## Getting Help

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Code of Conduct**: Please read our [Code of Conduct](CODE_OF_CONDUCT.md)

## Recognition

Contributors will be recognized in:
- Release notes
- GitHub contributors list

Thank you for contributing to the project! 🚀
