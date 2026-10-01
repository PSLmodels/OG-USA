# GNU Makefile that documents and automates common development operations
#              using the GNU make tool (version >= 3.81)
# Development is typically conducted on Linux or Max OS X (with the Xcode
#              command-line tools installed), so this Makefile is designed
#              to work in that environment (and not on Windows).
# USAGE: OG-USA$ make [TARGET]
#
# Requires `uv` (https://docs.astral.sh/uv/) to be installed.


.PHONY=help
help:
	@echo "USAGE: make [TARGET]"
	@echo "TARGETS:"
	@echo "help       : show help message"
	@echo "clean      : remove .pyc files, caches, and the local .venv"
	@echo "install    : create .venv and install ogusa with dev deps via uv sync"
	@echo "test       : run tests with coverage (matches CI)"
	@echo "pytest     : run tests with warnings suppressed"
	@echo "lint       : check formatting and lint with ruff (no changes)"
	@echo "coverage   : generate test coverage report"
	@echo "git-sync   : synchronize local, origin, and upstream Git repos"
	@echo "git-pr N=n : create local pr-n branch containing upstream PR"
	@echo "pip-package: build sdist + wheel via uv build"
	@echo "format     : format code with ruff and auto-fix lint issues"
	@echo "documentation : build documentation using jupyter-book"
	@echo "new-baseline : update baseline parameters and save to json file"

.PHONY=clean
clean:
	@find . -name '*.pyc' -exec rm {} \;
	@find . -name '__pycache__' -type d -exec rm -r {} +
	@find . -name '.pytest_cache' -type d -exec rm -r {} +
	@find . -name '.ruff_cache' -type d -exec rm -r {} +
	@rm -rf .venv build dist *.egg-info

install:
	uv sync --extra dev

test:
	uv run pytest -m 'not local' --cov=./ --cov-report=xml

.PHONY=pytest
pytest:
	@uv run pytest -W ignore

.PHONY=lint
lint:
	uv run ruff format --check .
	uv run ruff check .

define coverage-cleanup
rm -f .coverage htmlcov/*
endef

COVMARK = ""

OS := $(shell uname -s)

.PHONY=coverage
coverage:
	@$(coverage-cleanup)
	@uv run coverage run -m pytest -v -m $(COVMARK) > /dev/null
	@uv run coverage html --ignore-errors
ifeq ($(OS), Darwin) # on Mac OS X
	@open htmlcov/index.html
else
	@echo "Open htmlcov/index.html in browser to view report"
endif
	@$(pytest-cleanup)

.PHONY=git-sync
git-sync:
	@./gitsync

.PHONY=git-pr
git-pr:
	@./gitpr $(N)

pip-package:
	uv build

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check . --fix
	uv run linecheck . --fix

.PHONY: documentation
documentation:
	uv run --extra docs jupyter-book clean docs/book
	uv run --extra docs python docs/create_doc_figures.py
	uv run --extra docs jupyter-book build docs/book

new-baseline:
	uv run python ogusa/update_baseline.py
