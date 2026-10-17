PYTHON ?= python

.PHONY: install check test

install:
	$(PYTHON) -m pip install --no-deps -e .

check:
	$(PYTHON) tools/check_repository.py

test:
	$(PYTHON) -m pytest
