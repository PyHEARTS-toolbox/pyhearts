PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

.PHONY: help test coverage coverage-html

help:
	@echo "Targets:"
	@echo "  make test          Run unit tests"
	@echo "  make coverage      Run tests + coverage (terminal + html + xml)"
	@echo "  make coverage-html Run tests + HTML coverage only"

test:
	$(PYTEST)

coverage:
	$(PYTEST) --cov=pyhearts --cov-report=term-missing --cov-report=xml --cov-report=html

coverage-html:
	$(PYTEST) --cov=pyhearts --cov-report=html




