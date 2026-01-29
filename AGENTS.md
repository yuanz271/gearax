# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-29T14:08:58Z
**Commit:** bde786d
**Branch:** main

## OVERVIEW
Gearax is a small Python `src/`-layout package of JAX/Equinox ML utilities: config-carrying modules, ZIP-based model serialization, and a sharded training loop with Rich progress.

## STRUCTURE
```text
./
├── pyproject.toml          # Hatchling build + dependency groups
├── uv.lock                 # uv lockfile (do not edit manually)
├── .pre-commit-config.yaml # ruff lint/format hooks
├── src/
│   └── gearax/             # runtime package
└── tests/                  # pytest
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Config-aware model base | `src/gearax/modules.py` | `ConfModule` (OmegaConf `DictConfig` carried as `static=True`) |
| Save/load models | `src/gearax/modules.py` | ZIP archive: entries `conf` (YAML) + `pytree` (Eqx leaves) |
| Training loop + early stopping | `src/gearax/trainer.py` | `train(...)`, `Monitor` (Rich progress; sharding + donation) |
| Subclass registry factory mixin | `src/gearax/mixin.py` | `SubclassRegistryMixin` + `get_subclass()` |
| Tests | `tests/` | function-style pytest; no `conftest.py` |

## CODE MAP (HIGH-SIGNAL)
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `ConfModule` | class | `src/gearax/modules.py` | Abstract base for Equinox modules that carry config |
| `save_model` | fn | `src/gearax/modules.py` | Serialize config + params to ZIP |
| `load_model` | fn | `src/gearax/modules.py` | Deserialize ZIP into provided `klass` |
| `SubclassRegistryMixin` | class | `src/gearax/mixin.py` | Auto-register direct subclasses by `__name__` |
| `Monitor` | class | `src/gearax/trainer.py` | Early stopping + progress tracking |
| `train` | fn | `src/gearax/trainer.py` | Main training loop (sharding + jit + early stop) |

## CONVENTIONS (PROJECT-SPECIFIC)
- Python packaging is **src-layout**: importable package lives at `src/gearax/`.
- Dependencies:
  - Runtime deps are in `[project].dependencies`.
  - Dev deps use **PEP 735 dependency groups** (`[dependency-groups].dev`), not pip “extras”.
- Formatting/linting is via **pre-commit** hooks configured for ruff (`ruff-check --fix` + `ruff-format`).

## ANTI-PATTERNS (THIS PROJECT)
- Don’t follow docs that suggest `.[dev]` extras: `pyproject.toml` defines **dependency groups**, not extras.
- Don’t assume `gearax` re-exports APIs at the top level; `src/gearax/__init__.py` currently only defines `__version__`.

## COMMANDS
```bash
# Environment (locked)
uv sync --frozen

# Include dev group (pytest)
uv sync --frozen --group dev

# Tests
uv run pytest

# Build distributions (uv native)
uv build

# Lockfile hygiene (CI-friendly)
uv lock --check
```

## NOTES / GOTCHAS
- No CI workflows are present (`.github/workflows/` absent). If you add CI, mirror the repo’s uv + pytest flow.
