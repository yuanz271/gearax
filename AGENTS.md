# PROJECT GUIDE (AGENTS)

This file is for automated coding agents working in this repository.

## OVERVIEW
- Gearax is a small JAX/Equinox utilities package.
- Features: config-carrying modules, ZIP serialization, sharded training loop.
- Python package uses a src/ layout; import from `gearax.*`.

## CORE CONCEPTS
- `ConfModule` ties an `OmegaConf` config to an Equinox module.
- `save_model`/`load_model` store configs + parameters in a two-entry ZIP.
- `train` provides a sharded training loop with early stopping and monitoring.
- `SubclassRegistryMixin` supports name-based factory registration.

## GITHUB FLOW
- Branch from `main` for all work; no direct changes on `main`.
- Keep branches small and focused; open a pull request for review.
- Use pull requests for merging to `main`; avoid rebasing shared branches.
- Sync with `main` before final review (merge or rebase locally as needed).
- Ensure tests/lint pass before requesting review.

## GIT SAFETY
- Do not commit, push, merge, rebase, tag, or otherwise change git history without explicit permission.
- Run `git status` and `git diff` before proposing any commit.

## FILE SAFETY
- Never delete files or folders without explicit permission.

## DEPENDENCIES & CONFIGURATION
- Do not add, remove, or upgrade dependencies without informing the user.
- Do not modify environment files (.env), credentials, secrets, or auth configs without permission.

## DOCUMENTATION
- Update relevant docs (README, inline docs, docstrings) whenever changes are made.
- Avoid creating new documentation files unless explicitly requested.

## REPO LAYOUT
```text
./
├── pyproject.toml          # Hatchling build + dependency groups
├── uv.lock                 # uv lockfile (do not edit manually)
├── .pre-commit-config.yaml # ruff lint + format
├── src/
│   └── gearax/
│       ├── __init__.py
│       ├── modules.py      # ConfModule + save/load
│       ├── trainer.py      # train() + Monitor
│       └── mixin.py        # SubclassRegistryMixin
└── tests/                  # pytest (function-style only)
```

## KEY FILES
- `src/gearax/modules.py`: ConfModule, save/load ZIP serialization.
- `src/gearax/trainer.py`: sharded training loop + early stopping.
- `src/gearax/mixin.py`: subclass registry pattern.
- `tests/`: pytest coverage for modules, mixins, trainer.

## WHERE TO LOOK
- Config-aware model base: `src/gearax/modules.py` (`ConfModule`).
- Serialization format: `src/gearax/modules.py` (`save_model`/`load_model`).
- Training loop + early stopping: `src/gearax/trainer.py` (`train`, `Monitor`).
- Subclass registry factory: `src/gearax/mixin.py` (`SubclassRegistryMixin`).
- Tests: `tests/test_*.py` (function-style).

## SERIALIZATION FORMAT
- ZIP archive with two entries:
  - `conf`: YAML from `OmegaConf.to_yaml`.
  - `pytree`: Equinox leaves via `eqx.tree_serialise_leaves`.
- Missing `conf` or `pytree` should raise `KeyError`.

## TRAINING LOOP CONTRACTS
- `train(...)` expects `dataloader(train_set, batch_size, max_epoch, key)`.
- The dataloader must yield `(batch, epoch, batch_in_epoch)`.
- Sharding is applied via `eqx.filter_shard` for model, optimizer state, and batches.
- JIT boundaries are explicit; be careful when moving sharding outside/inside JIT.

## COMMANDS (BUILD/LINT/TEST)
Environment setup:
```bash
uv sync --frozen
uv sync --frozen --group dev
```

Run tests:
```bash
uv run pytest
```

Run a single test file:
```bash
uv run pytest tests/test_trainer.py
```

Run a single test function:
```bash
uv run pytest tests/test_trainer.py::test_train_smoke_single_device_sharding
```

Lint/format (preferred):
```bash
pre-commit run -a
```

Build distributions:
```bash
uv build
```

Lockfile check:
```bash
uv lock --check
```

## CODE STYLE
General:
- Python 3.11+ only; use modern type syntax (`X | Y`).
- Ruff handles formatting; do not hand-format beyond it.
- Docstrings use a Numpy-style structure with Parameters/Returns.

Imports:
- Order: standard library, third-party, local imports.
- Keep imports explicit; no wildcard imports.
- Prefer module-level imports; local imports only to avoid heavy deps in tests.

Types:
- Type hints are expected on public functions and class attributes.
- Use `jax.Array` or `jax.typing.ArrayLike` for array inputs where appropriate.
- In tests, lightweight typing is fine; clarity over completeness.

Naming:
- Functions/variables: `snake_case`.
- Classes: `CamelCase`.
- Constants: `UPPER_SNAKE_CASE` if introduced.
- Tests: `test_*` function names; no test classes.

Formatting:
- Use `ruff-format` via pre-commit; keep line length consistent with Ruff.
- One statement per line; avoid backslash continuations.

Error handling:
- Use `ValueError` for invalid arguments or missing registry entries.
- Use `KeyError` when ZIP entries are missing (matches existing tests).
- Prefer `pytest.raises(..., match=...)` for failure-path tests.

JAX/EQX conventions:
- Keep functions pure where possible; avoid side effects in training steps.
- Be explicit about sharding (via `eqx.filter_shard`) and JIT boundaries.
- `eqx.filter_jit(donate="all")` means inputs are consumed; do not reuse.
- `load_model(...)` runs `eqx.filter_eval_shape(..., None)`; model init must tolerate `key=None`.

## TESTING CONVENTIONS
- Tests are function-style only (`def test_*():`).
- No shared fixtures currently (`conftest.py` absent).
- Use deterministic `jax.random.key(0)` or `jr.key(0)` in tests.

## PROJECT GOTCHAS
- `gearax.__init__` only defines `__version__`; no re-exports.
- Dependency groups are in `[dependency-groups]`, not pip extras.
- `Monitor.step()` calls `.item()` on validation loss (host sync).

## EDITING NOTES
- Keep changes minimal and focused; prefer small diffs.
- Avoid adding docs-only files under `src/gearax/` unless they must ship.
- Add comments only when behavior is non-obvious.
- Preserve existing public APIs and function signatures unless required.
- When touching training code, keep pure-function style and avoid side effects.
- Update relevant docs (README, inline docs, docstrings) whenever changes are made.

## AGENT-SPECIFIC NOTES
- This repo includes additional agent notes in:
  - `src/gearax/AGENTS.md`
  - `tests/AGENTS.md`
- No Cursor rules found in `.cursor/rules/` or `.cursorrules`.
- No Copilot rules found in `.github/copilot-instructions.md`.
