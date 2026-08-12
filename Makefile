PYTHON ?= python3.11
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
RUFF_TARGETS := app system scripts tests debugger.py makeJob.py

.PHONY: setup-dev lint lint-author format format-author format-check compile test check clean-dev

setup-dev: $(RUFF)

$(RUFF): requirements-dev.txt pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt

lint: $(RUFF)
	$(RUFF) check $(RUFF_TARGETS)

lint-author: $(RUFF)
	$(RUFF) check app/code/computation

format: $(RUFF)
	$(RUFF) check --fix $(RUFF_TARGETS)
	$(RUFF) format $(RUFF_TARGETS)

format-author: $(RUFF)
	$(RUFF) check --fix app/code/computation
	$(RUFF) format app/code/computation

format-check: $(RUFF)
	$(RUFF) format --check $(RUFF_TARGETS)

compile: $(RUFF)
	PYTHONPATH=app/code $(VENV_PYTHON) -m compileall -q app system scripts tests debugger.py makeJob.py

test:
	PYTHONPATH=app/code $(VENV_PYTHON) -m unittest discover -s tests

check: lint format-check compile test

clean-dev:
	rm -rf $(VENV) .ruff_cache
