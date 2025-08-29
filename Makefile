#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER=uv run python3
ifneq (,$(wildcard ./.env))
    -include .env
    export
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf *.egg-info
	rm -rf **/__pycache__
	rm -rf **.ipynb_checkpoints
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f coverage.xml
	rm -f .coverage
	rm -rf reports
	rm -rf public
	rm -rf docs/generated

## Check commit message with pre-commit
.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

## Check typing with mypy
.PHONY: type-check
type-check:
	mypy risk_control tests examples --config-file=pyproject.toml

## Lint using ruff
.PHONY: lint
lint:
	ruff check --config pyproject.toml

## Lint using ruff
.PHONY: format
format:
	ruff check --select I --select RUF022 --fix
	ruff format --config pyproject.toml

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Run tests and coverage test on package
.PHONY: tests
tests:
	@echo ">>> Running coverage tests at root directory..."
	$(PYTHON_INTERPRETER) -m coverage run -m pytest
	@echo ">>> Coverage tests finished."

## Run tests and coverage report
.PHONY: report-tests
report-tests:
	@echo ">>> Compiling report..."
	$(PYTHON_INTERPRETER) -m coverage report --ignore-errors
	$(PYTHON_INTERPRETER) -m coverage html --ignore-errors
	$(PYTHON_INTERPRETER) -m coverage xml --ignore-errors -o coverage.xml
	@echo ">>> Reports compiled."

## Clean the test/coverage caches
.PHONY: clean-tests
clean-tests:
	@echo ">>> Removing coverage caches..."
	@echo ">>> ... at: htmlcov and .coverage"
	rm -rf htmlcov .coverage .pytest_cache coverage.xml
	@echo ">>> Caches removed."

#################################################################################
# DOCUMENTATION                                                                 #
#################################################################################

## Build the documentation
.PHONY: docs
docs:
	@echo ">>> Building documentation..."
	mkdocs build

## Serve the documentation
.PHONY: serve
serve:
	@echo ">>> Serving documentation..."
	mkdocs serve

## Clean the documentation
.PHONY: clean-docs
clean-docs:
	@echo ">>> Removing documentation..."
	@echo ">>> ... at: site"
	rm -rf site
	rm -rf docs/generated
	@echo ">>> Documentation removed."

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:35}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
