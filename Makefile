# AutoML Studio Pro - Development Makefile
# =========================================

.PHONY: help install install-dev run test lint format clean docker-build docker-run

# Default target
help:
	@echo "AutoML Studio Pro - Development Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  setup         Complete development setup"
	@echo ""
	@echo "Development Commands:"
	@echo "  run           Run the Streamlit application"
	@echo "  test          Run all tests with coverage"
	@echo "  test-fast     Run tests without coverage"
	@echo "  lint          Run linting with Ruff"
	@echo "  format        Format code with Ruff"
	@echo "  type-check    Run type checking with mypy"
	@echo "  security      Run security scan with Safety"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run Docker container"
	@echo "  docker-up     Start with Docker Compose"
	@echo "  docker-down   Stop Docker Compose"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  clean         Remove cache and temp files"
	@echo "  clean-all     Remove all generated files"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs          Generate documentation"
	@echo "  serve-docs    Serve documentation locally"

# ==========================================
# Setup Commands
# ==========================================

install:
	@echo "Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

setup: install-dev
	@echo "Setting up development environment..."
	python -m pytest --version || echo "pytest installed"
	@echo "Setup complete! Run 'make run' to start the app."

# ==========================================
# Development Commands
# ==========================================

run:
	@echo "Starting AutoML Studio Pro..."
	streamlit run app.py

test:
	@echo "Running tests with coverage..."
	python -m pytest tests/ -v --cov=automl_app --cov-report=html --cov-report=term

test-fast:
	@echo "Running tests..."
	python -m pytest tests/ -v

lint:
	@echo "Running linting..."
	ruff check automl_app/ app.py tests/

lint-fix:
	@echo "Running linting with auto-fix..."
	ruff check --fix automl_app/ app.py tests/

format:
	@echo "Formatting code..."
	ruff format automl_app/ app.py tests/

type-check:
	@echo "Running type checking..."
	mypy automl_app/ --ignore-missing-imports

security:
	@echo "Running security scan..."
	safety check -r requirements.txt
	safety check -r requirements-dev.txt

check: lint format test
	@echo "All checks passed!"

# ==========================================
# Docker Commands
# ==========================================

docker-build:
	@echo "Building Docker image..."
	docker build -t automl-studio-pro .

docker-run:
	@echo "Running Docker container..."
	docker run -d -p 8501:8501 --name automl_studio_pro automl-studio-pro

docker-up:
	@echo "Starting with Docker Compose..."
	docker compose up -d --build

docker-down:
	@echo "Stopping Docker Compose..."
	docker compose down

docker-logs:
	@echo "Showing Docker logs..."
	docker logs automl_studio_pro

# ==========================================
# Cleanup Commands
# ==========================================

clean:
	@echo "Cleaning cache and temp files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .tmp/ 2>/dev/null || true
	@echo "Clean complete!"

clean-all: clean
	@echo "Removing all generated files..."
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	rm -rf artifacts/*.joblib 2>/dev/null || true
	rm -rf artifacts/*.json 2>/dev/null || true
	@echo "Deep clean complete!"

# ==========================================
# Documentation Commands
# ==========================================

docs:
	@echo "Documentation is in the docs/ directory"
	@echo "Current documentation files:"
	@find docs/ -type f -name "*.md" 2>/dev/null || echo "No docs found"

serve-docs:
	@echo "Serving documentation..."
	@echo "Open docs/ folder to view documentation files"

# ==========================================
# Git Commands
# ==========================================

git-status:
	git status

git-log:
	git log --oneline -10

git-push:
	git add -A
	git commit -m "Update: $(msg)"
	git push origin main

# ==========================================
# Utility Commands
# ==========================================

version:
	@echo "AutoML Studio Pro v1.3.0"

check-deps:
	@echo "Checking for outdated dependencies..."
	pip list --outdated

update-deps:
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt
	pip freeze > requirements.txt
