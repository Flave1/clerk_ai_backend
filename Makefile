# AI Receptionist Backend Makefile

.PHONY: help install dev test test-unit test-integration test-e2e lint format clean build run-api run-workers run-rt-gateway setup-test-db run-meeting-agent test-meeting-agent

# Default target
help:
	@echo "Available commands:"
	@echo "  install         - Install dependencies"
	@echo "  dev            - Install dev dependencies"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e       - Run end-to-end tests only"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  clean          - Clean up generated files"
	@echo "  build          - Build Docker images"
	@echo "  run-api        - Run API service"
	@echo "  run-workers    - Run background workers"
	@echo "  run-rt-gateway - Run real-time gateway"
	@echo "  setup-test-db  - Setup test database"
	@echo "  create-tables  - Create DynamoDB tables"
	@echo "  run-meeting-agent - Run meeting agent service"
	@echo "  test-meeting-agent - Run meeting agent tests"

# Installation
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt

# Testing
test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

test-coverage:
	pytest tests/ --cov=services --cov=shared --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -x --tb=short

test-parallel:
	pytest tests/ -v -n auto

# Code Quality
lint:
	flake8 services/ shared/ tests/
	mypy services/ shared/
	isort --check-only services/ shared/ tests/

format:
	black services/ shared/ tests/
	isort services/ shared/ tests/

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Development
run-api:
	PYTHONPATH=. python -m services.api.main

run-workers:
	PYTHONPATH=. python -m services.workers.worker

run-rt-gateway:
	PYTHONPATH=. python -m services.rt_gateway.app

# Meeting Agent
run-meeting-agent:
	PYTHONPATH=. python -m services.meeting_agent.main

test-meeting-agent:
	PYTHONPATH=. pytest tests/integration/meeting_agent/ -v -m integration

# Database
setup-test-db:
	PYTHONPATH=. python tests/setup_test_db.py

create-tables:
	PYTHONPATH=. python scripts/create_tables.py create

delete-tables:
	PYTHONPATH=. python scripts/create_tables.py --delete

# See detailed errors
run-integration-tests:
#PYTHONPATH=. pytest tests/integration/ -v -m integration
#PYTHONPATH=. pytest tests/integration/ -v -s --tb=long
	PYTHONPATH=. pytest tests/integration/api/test_conversation_workflow.py::TestConversationWorkflow::test_create_conversation_and_add_turns -v -s --tb=long

# Docker
build:
	docker build -f infra/docker/api.Dockerfile -t clerk-api .
	docker build -f infra/docker/workers.Dockerfile -t clerk-workers .
	docker build -f infra/docker/rt_gateway.Dockerfile -t clerk-rt-gateway .

build-dev:
	docker-compose -f docker-compose.dev.yml build

run-docker:
	docker-compose up

run-docker-dev:
	docker-compose -f docker-compose.dev.yml up

# Testing with Docker
test-docker:
	docker-compose -f docker-compose.test.yml run --rm api pytest tests/

# Production
deploy-staging:
	@echo "Deploying to staging..."
	# Add staging deployment commands here

deploy-prod:
	@echo "Deploying to production..."
	# Add production deployment commands here

# Monitoring
logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-workers:
	docker-compose logs -f workers

# Database management
db-migrate:
	@echo "Running database migrations..."
	# Add migration commands here

db-seed:
	@echo "Seeding database..."
	# Add seed commands here

# Environment
env-example:
	cp .env.example .env

env-check:
	@echo "Checking environment variables..."
	@python -c "from shared.config import get_settings; print('Environment check passed')"

# Health checks
health-check:
	@echo "Running health checks..."
	curl -f http://localhost:8000/health || exit 1

# Documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here

# Security
security-check:
	@echo "Running security checks..."
	bandit -r services/ shared/

# Performance
perf-test:
	@echo "Running performance tests..."
	pytest tests/performance/ -v

# Backup
backup-db:
	@echo "Backing up database..."
	# Add backup commands here

# Restore
restore-db:
	@echo "Restoring database..."
	# Add restore commands here