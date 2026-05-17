# Pymerag Makefile
.PHONY: help setup up down test lint eval docs clean dataset

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies with uv
	uv sync --group dev

up: ## Start all services
	docker compose up -d

down: ## Stop all services
	docker compose down

test: ## Run test suite
	uv run pytest -v

lint: ## Run linter
	uv run ruff check .

format: ## Auto-format code
	uv run ruff check --fix .
	uv run ruff format .

eval: ## Run RAGAS evaluation
	uv run python eval/run_ragas.py

docs: ## Serve API docs locally
	uv run uvicorn app.main:app --reload --port 8000

dataset: ## Generate synthetic dataset
	uv run python scripts/generate_dataset.py

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .venv/ dist/ build/ *.egg-info/
