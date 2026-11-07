UV ?= uv

.PHONY: install sync test test-unit lint format build

install: sync

sync:
	$(UV) sync --extra test

test: sync
	$(UV) run --extra test pytest

test-unit: sync
	$(UV) run --extra test pytest -m 'not integration'

lint: sync
	$(UV) run --extra test ruff check .
	$(UV) run --extra test ruff format --check .

format: sync
	$(UV) run --extra test ruff format .

build:
	$(UV) build
