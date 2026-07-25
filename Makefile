# Dota Analyst — Makefile
# Run `make help` for the list of targets.

SHELL := /usr/bin/env bash

PYTHON ?= python
VENV ?= .venv
PORT  ?= 8000

ifeq ($(OS),Windows_NT)
	VENV_PY := $(VENV)/Scripts/python.exe
else
	VENV_PY := $(VENV)/bin/python
endif

# ----------------------------------------------------------------------
# Help
# ----------------------------------------------------------------------

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

.PHONY: venv
venv: ## Create virtualenv
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip

.PHONY: install
install: ## Install runtime + dev dependencies
	$(VENV_PY) -m pip install -e ".[dev]"

.PHONY: install-prod
install-prod: ## Install runtime only
	$(VENV_PY) -m pip install -e .

.PHONY: clean
clean: ## Remove build / cache artefacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ----------------------------------------------------------------------
# Quality gates
# ----------------------------------------------------------------------

.PHONY: test
test: ## Run pytest
	$(VENV_PY) -m pytest tests/ -v

.PHONY: test-cov
test-cov: ## Run pytest with coverage
	$(VENV_PY) -m pytest tests/ --cov=business --cov=gateway --cov-report=term-missing

.PHONY: compile
compile: ## Syntax-check every Python file
	$(VENV_PY) -m compileall -q business/ gateway/ tests/

.PHONY: ci
ci: compile test ## Run CI-equivalent locally (compile + tests)

# ----------------------------------------------------------------------
# Run (single-process — for development against one service at a time)
# ----------------------------------------------------------------------

.PHONY: run-business
run-business: ## Run the business service on :8000
	$(VENV_PY) -m uvicorn business.app:app --reload --host 0.0.0.0 --port 8000

.PHONY: run-gateway
run-gateway: ## Run the gateway on :8000 (talks to business on :8001)
	BUSINESS_URL=http://localhost:8001 $(VENV_PY) -m uvicorn gateway.app:app --reload --host 0.0.0.0 --port 8000

.PHONY: run-legacy
run-legacy: ## Run the legacy monolith (0.0.x — single-port, no auth)
	$(VENV_PY) -m uvicorn business.app:app --reload --host 0.0.0.0 --port 8000
	# Note: business.app no longer serves / — use docker compose for full stack

# ----------------------------------------------------------------------
# Docker (preferred way to run the full 0.1.0 stack)
# ----------------------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build all three service images (web, gateway, business)
	docker compose build

.PHONY: docker-up
docker-up: ## Start the full stack (web, gateway, business) in foreground
	docker compose up

.PHONY: docker-up-d
docker-up-d: ## Start the full stack detached
	docker compose up -d

.PHONY: docker-down
docker-down: ## Stop the stack
	docker compose down

.PHONY: docker-logs
docker-logs: ## Tail logs from all services
	docker compose logs -f

.PHONY: docker-ps
docker-ps: ## Show running containers
	docker compose ps

.PHONY: docker-shell
docker-shell: ## Shell into the business container
	docker compose exec business sh

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

.PHONY: data-stats
data-stats: ## Print stats about ml_data/full_matches/
	@echo "ml_data/full_matches/:"
	@ls ml_data/full_matches/ 2>/dev/null | wc -l | xargs -I{} echo "  files: {}"
	@du -sh ml_data/full_matches/ 2>/dev/null | awk '{print "  size:  " $$1}'

.PHONY: collect-tier1
collect-tier1: ## Run the DatDota tier-1 collector (uses 500 req/day budget!)
	$(VENV_PY) -m business.datdota_client collect_all_tier1_matches

# ----------------------------------------------------------------------
# Default
# ----------------------------------------------------------------------

.DEFAULT_GOAL := help
