# AGENTS.md
Guidance for coding agents working in **AutoML Studio Pro**.
## 1) Repository Overview
- Language: Python (3.9+). CI matrix: 3.11/3.12/3.13.
- App type: Streamlit UI + sklearn/ML helpers.
- Entry point: `app.py`.
- Core logic: `automl_app/core/`.
- UI tabs: `automl_app/ui/tabs/`.
- Tests: `tests/`.
- CI workflows: `.github/workflows/{ci,tests,cd}.yml`.
## 2) Environment Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```
Run app:
```bash
streamlit run app.py
# or
make run
```
## 3) Build / Lint / Test Commands
Preferred pre-commit checks:
```bash
ruff check .
python -m compileall app.py automl_app tests
pytest -q
safety check -r requirements.txt
safety check -r requirements-dev.txt
```
Makefile shortcuts:
```bash
make lint        # ruff check automl_app/ app.py tests/
make format      # ruff format automl_app/ app.py tests/
make test        # pytest with coverage
make test-fast   # pytest without coverage
make type-check  # mypy automl_app/ --ignore-missing-imports
make security    # safety check requirements files
```
### Running a single test (important)
Use pytest node IDs:
```bash
# Single test file
pytest tests/test_helpers.py -q

# Single test function
pytest tests/test_helpers.py::test_quick_dtype_buckets_excludes_target -q

# Class method style (if class tests are introduced)
pytest tests/test_file.py::TestClass::test_method -q
```
Other useful selectors:
```bash
pytest -k "profile_dataset and not slow" -q
pytest -m unit -q
pytest -m "not slow" -q
```
Coverage run:
```bash
pytest -q --cov=. --cov-report=xml --cov-report=term
```
## 4) Formatting and Linting Rules
From `pyproject.toml`:
- Tooling: Ruff (lint + format).
- Line length: 100.
- Target: py39.
- Import sort: Ruff/isort (`automl_app` as first-party).
- Enabled lint families: `E,W,F,I,B,C4,UP,S,T20,SIM,RUF`.
- Notable ignores: `E501`, `S101`, `T201`.

Agent expectations:
1. Run `ruff check .` after edits.
2. Run `ruff format` on touched Python files.
3. Keep diffs minimal and task-focused.
## 5) Python Style Guidelines
### Imports
- Group in order: stdlib, third-party, local.
- Prefer explicit imports; avoid wildcard imports.
- Remove unused imports.

### Types
- Add type hints for new/changed functions.
- Prefer concrete return types (`dict[str, float]`, `tuple[...]`).
- Keep existing signatures stable unless required.

### Naming
- Functions/variables: `snake_case`.
- Constants: `UPPER_SNAKE_CASE`.
- Tests: `test_<behavior>`.
- Predicates: `is_*`, `has_*`, `can_*`.

### Function design
- Prefer small, single-purpose functions.
- Prefer pure transformations in core logic.
- Avoid hidden mutations/global state.
- Use early returns to reduce nesting.

### Streamlit-specific
- Give unique `key=` when widget collisions are possible.
- Use `st.session_state` intentionally; initialize keys defensively.
- Keep heavy ML/business logic outside UI rendering blocks.

## 6) Error Handling, Logging, Security
- Validate at boundaries (user input, uploaded files, artifacts).
- Avoid bare `except:`; catch specific exceptions.
- Do not silently swallow exceptions.
- Prefer `logging` in non-UI logic.
- Never use `eval` on untrusted input; use safe parsing (`ast.literal_eval`).
- Keep artifact/schema writes deterministic and UTF-8.
- Security scanning is requirements-file based (Safety).

## 7) Testing Standards
- Follow AAA: Arrange → Act → Assert.
- Test happy path + edge cases + error paths.
- Add regression tests for bug fixes.
- Keep tests deterministic and independent.
- After non-trivial changes, run:
  - touched test(s),
  - related module tests,
  - full `pytest -q` before handoff.

## 8) Documentation Expectations
- Update docs when behavior/commands change (`README.md`, `docs/`, changelog).
- Write comments for **why**, not obvious **what**.
- Keep commands copy-pasteable and cross-platform where possible.

## 9) Cursor / Copilot Rules Status
Checked requested locations:
- `.cursorrules`
- `.cursor/rules/`
- `.github/copilot-instructions.md`

Current status in this repository: **not found**.
If these files are added later, treat them as high-priority instructions.

## 10) Pre-PR Agent Checklist
1. `ruff check .`
2. `python -m compileall app.py automl_app tests`
3. `pytest -q` (targeted + full run for larger changes)
4. `safety check -r requirements.txt`
5. `safety check -r requirements-dev.txt`
6. Update docs/tests for behavior changes
7. Confirm no secrets/generated artifacts are accidentally staged
This document is intended to keep autonomous agent changes consistent, safe, and reviewable.
