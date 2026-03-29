# Installation Guide

This guide covers all methods to install and run AutoML Studio Pro.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation Methods](#installation-methods)
  - [Standard Installation](#standard-installation)
  - [Development Installation](#development-installation)
  - [Docker Installation](#docker-installation)
  - [Streamlit Cloud](#streamlit-cloud)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Runtime environment |
| pip | 21.0+ | Package manager |

### Optional

| Software | Version | Purpose |
|----------|---------|---------|
| Git | 2.30+ | Source code management |
| Docker | 20.10+ | Container deployment |
| Make | 4.0+ | Development commands |

---

## Quick Start

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/himanshu231204/AutoML-Studio-Pro-.git
cd AutoML-Studio-Pro-

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Installation Methods

### Standard Installation

For end users who want to run the application:

```bash
# 1. Clone repository
git clone https://github.com/himanshu231204/AutoML-Studio-Pro-.git
cd AutoML-Studio-Pro-

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install production dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app.py
```

### Development Installation

For contributors and developers:

```bash
# 1. Clone and enter directory
git clone https://github.com/himanshu231204/AutoML-Studio-Pro-.git
cd AutoML-Studio-Pro-

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install all dependencies (production + development)
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Install package in editable mode
pip install -e .

# 5. Setup pre-commit hooks (optional but recommended)
pre-commit install

# 6. Run tests to verify setup
make test

# 7. Run application
make run
```

#### Using Make Commands

After development installation, use these commands:

| Command | Description |
|---------|-------------|
| `make run` | Start the application |
| `make test` | Run tests with coverage |
| `make lint` | Run linting |
| `make format` | Format code |
| `make check` | Run all checks (lint + format + test) |
| `make clean` | Clean cache files |

### Docker Installation

#### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/himanshu231204/AutoML-Studio-Pro-.git
cd AutoML-Studio-Pro-

# Build and run
docker compose up -d --build

# Access at http://localhost:8501

# Stop
docker compose down
```

#### Option 2: Docker CLI

```bash
# Build image
docker build -t automl-studio-pro .

# Run container
docker run -d -p 8501:8501 --name automl_studio_pro automl-studio-pro

# Access at http://localhost:8501

# Stop and remove
docker stop automl_studio_pro
docker rm automl_studio_pro
```

#### Docker Environment Variables

You can customize the Docker container with environment variables:

```bash
docker run -d \
  -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  --name automl_studio_pro \
  automl-studio-pro
```

### Streamlit Cloud

Deploy to Streamlit Community Cloud (free):

1. Fork the repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your forked repository
5. Set main file path: `app.py`
6. Click "Deploy!"

The app will be available at `https://your-app-name.streamlit.app`

---

## Verification

After installation, verify everything works:

```bash
# Check Python version
python --version  # Should be 3.9+

# Check installed packages
pip list | grep -E "streamlit|scikit-learn|pandas"

# Run the application
streamlit run app.py

# Run tests (development)
pytest tests/ -v
```

---

## Troubleshooting

### Common Issues

#### "Python not found"

**Solution**: Ensure Python 3.9+ is installed and in your PATH.

```bash
python3 --version  # Try python3 instead
```

#### "pip not found"

**Solution**: Use `python -m pip` instead of `pip`.

```bash
python -m pip install -r requirements.txt
```

#### "Port 8501 already in use"

**Solution**: Use a different port.

```bash
streamlit run app.py --server.port 8502
```

#### "Module not found" errors

**Solution**: Ensure virtual environment is activated.

```bash
# Check if venv is active (should show venv path)
which python  # macOS/Linux
where python  # Windows

# If not, activate it:
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

#### "SHAP installation failed"

**Solution**: SHAP requires build tools. Use pre-built wheel:

```bash
pip install shap --prefer-binary
```

Or skip SHAP (app works without it):

```bash
pip install -r requirements.txt --ignore-installed shap
```

#### Docker: "Permission denied"

**Solution**: Run with sudo (Linux) or ensure Docker Desktop is running.

```bash
sudo docker compose up -d --build
```

### Getting Help

If you encounter issues not listed here:

1. Check [FAQ](FAQ.md)
2. Search [existing issues](https://github.com/himanshu231204/AutoML-Studio-Pro-/issues)
3. [Create a new issue](https://github.com/himanshu231204/AutoML-Studio-Pro-/issues/new)

---

## Next Steps

- Read the [Usage Guide](USAGE.md) to learn how to use the application
- Check [Architecture](../ARCHITECTURE.md) for technical details
- See [Contributing Guide](../CONTRIBUTING.md) to contribute to the project

---

**Last Updated:** March 2026
