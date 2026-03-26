# Contributing Guide

Thanks for contributing to AutoML Studio Pro.

## Development Setup

1. Fork and clone the repository.
2. Create a virtual environment.
3. Install dependencies.
4. Run the app and tests locally.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
pytest -q
```

## Branch Strategy

- Use feature branches from `main`
- Branch name format: `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>`

## Commit Guidelines

- Write clear, scoped commit messages
- Keep commits focused and reviewable
- Reference issues where applicable (example: `Fixes #23`)

## Pull Request Checklist

Before opening a PR:

- Ensure tests pass locally
- Add or update tests for behavioral changes
- Update docs if user-facing behavior changed
- Keep PR scope limited to one concern

## Coding Standards

- Follow existing project style and module organization
- Prefer small reusable functions over large blocks
- Add concise comments only when logic is non-obvious

## Reporting Bugs and Requesting Features

Use the issue templates in `.github/ISSUE_TEMPLATE`:

- Bug report template for reproducible bugs
- Feature request template for enhancements

## Review Expectations

- Be respectful and constructive
- Discuss trade-offs with evidence
- Prioritize correctness, maintainability, and user impact

## Code of Conduct

By participating, you agree to follow the project Code of Conduct in `CODE_OF_CONDUCT.md`.
