# Repository Guidelines

## Project Structure & Module Organization
Gearax is packaged as a standard Python project with all runtime code under `src/gearax/`. Core components live in `modules.py` (configuration-aware Equinox modules), `mixin.py` (subclass registry utilities), and `trainer.py` (training loop helpers and Rich progress integration). Public APIs are re-exported through `__init__.py`. Tests reside in `tests/` alongside fixtures; mirror the runtime module layout when adding scenarios. The lockfile `uv.lock` keeps dependency resolution reproducible—update it only when intentionally upgrading requirements.

## Build, Test, and Development Commands
Use `uv` to stay consistent with the existing toolchain:
- `uv sync --frozen` installs the exact dependency set from `uv.lock`.
- `uv run python -m pip install -e .[dev]` adds an editable install with test extras for local iteration.
- `uv run pytest` executes the full test suite; append `-k <pattern>` to focus on a single module.
- `uv run python -m build` produces a wheel via Hatchling; artifacts land in `dist/`.
Prefer `uv run <cmd>` over raw `python` to ensure the locked interpreter and packages are used.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive snake_case for functions, and CapWords for classes. Maintain type hints and numpy-style docstrings, matching the existing modules. Keep configuration-carrying classes immutable where possible (e.g., declare `DictConfig` fields with `static=True`). When introducing new utilities, expose them through `src/gearax/__init__.py` to preserve the public surface.

## Testing Guidelines
Author tests with `pytest`, placing files as `tests/test_<feature>.py`. Mirror class or function names in test names (e.g., `test_conf_module_roundtrip`). Include round-trip serialization checks when touching `modules.py` and deterministic behavior checks when randomness is involved—seed with `jax.random.PRNGKey`. New features should have positive and failure-path assertions before requesting review.

## Commit & Pull Request Guidelines
Commits in this repository use short, imperative summaries (e.g., `fix argument donation`); keep them under 72 characters and scope-focused. For pull requests, provide: (1) a concise description of the change and motivation, (2) references to any linked issues, and (3) a bullet list of validation steps (`uv run pytest`, sample scripts, etc.). Include screenshots or Rich output captures only if the change affects CLI progress reporting. Request review once CI is green and TODOs are cleared.
