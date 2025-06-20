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

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Check typing with mypy
.PHONY: type-check
type-check:
	mypy risk_control tests examples

## Lint using flake8
.PHONY: lint-flake
lint-flake:
	flake8 risk_control tests examples

## Lint using black
.PHONY: lint-black
lint-black:
	black --check --config pyproject.toml risk_control tests examples

## Lint using isort
.PHONY: lint-isort
lint-isort:
	isort --check --diff --profile black --resolve-all-configs risk_control tests examples

## Lint using flake8, isort and black (use `make format` to do formatting)
.PHONY: lint
lint:
	$(MAKE) lint-flake
	$(MAKE) lint-black
	$(MAKE) lint-isort

## Format source code with isort and black
.PHONY: format
format:
	isort --profile black --resolve-all-configs risk_control tests examples
	black --config pyproject.toml risk_control tests examples

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
## Clean xml test/coverage caches
.PHONY: clean-tests
clean-tests:
	@echo ">>> Removing coverage caches..."
	@echo ">>> ... at: htmlcov and .coverage"
	rm -rf htmlcov .coverage .pytest_cache
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
