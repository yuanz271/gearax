# src/gearax/ — Runtime Package

## OVERVIEW
Runtime JAX/Equinox utilities (models + serialization + training). This directory should remain importable as `gearax`.

## BOUNDARIES
- Put runtime/library code here.
- Put tests in `tests/` (no `tests/` under `src/gearax/`).
- Avoid adding “docs-only” files that must ship in the wheel unless intentional (this `AGENTS.md` is the exception).

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Config-carrying model base | `modules.py` | `ConfModule` carries `DictConfig` as `static=True` |
| Serialization format | `modules.py` | ZIP entries: `conf` (YAML) + `pytree` (Eqx leaves) |
| Training loop | `trainer.py` | `train(...)` + `Monitor` (early stop + progress) |
| Registry/factory mixin | `mixin.py` | `SubclassRegistryMixin.get_subclass(name)` |
| Package surface | `__init__.py` | Defines `__version__` only; no re-exports |

## GOTCHAS (HIGH-SIGNAL)
- `load_model(...)` uses `eqx.filter_eval_shape(klass, ..., None)` → concrete model constructors must tolerate `key=None` (or ignore the key) during shape evaluation.
- `train(...)` assumes a `dataloader(train_set, batch_size, max_epoch, loader_key)` yielding `(batch, epoch, batch_in_epoch)`; this contract is implicit but required.
- `Monitor.step()` calls `.item()` on the validation loss, which synchronizes device→host each validation step.
- `@eqx.filter_jit(donate="all")` in `train_step` means callers must treat inputs as consumed and only use returned values.
