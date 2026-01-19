# gearax/ — ML Utilities Package

## OVERVIEW

JAX/Equinox utilities: config-aware modules, model serialization, training helpers.

## STRUCTURE

| Module | Purpose |
|--------|---------|
| `modules.py` | `ConfModule` base, `save_model()`, `load_model()` |
| `mixin.py` | `SubclassRegistryMixin` for factory patterns |
| `trainer.py` | Training infrastructure (if extended) |

## WHERE TO LOOK

| Task | Start Here |
|------|------------|
| Create configurable model | Subclass `ConfModule` |
| Add subclass auto-registration | Use `SubclassRegistryMixin` |
| Save/load model with config | `save_model()`, `load_model()` |

## CONVENTIONS

- `ConfModule`: combines `eqx.Module` with `OmegaConf.DictConfig`
- Serialization: ZIP archive with params + config YAML
- `SubclassRegistryMixin`: auto-registers subclasses by `__name__`

## NOTES

- Editable dep of main jaxfads: run `uv sync` after changes
- Tests in `gearax/tests/`
