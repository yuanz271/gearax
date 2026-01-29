# Gearax

Gearax is a small set of utilities for training JAX + Equinox models with:
- **Config-carrying modules** (OmegaConf `DictConfig` stored alongside parameters)
- **ZIP-based model serialization** (config + pytree leaves)
- **A sharded training loop** with early stopping and Rich progress

This is a library package (no CLI).

## Install

This repo is set up for **uv** (recommended) and a standard editable install.

```bash
# Locked environment (runtime deps)
uv sync --frozen

# With dev deps (pytest)
uv sync --frozen --group dev

# Editable install (if you prefer pip)
pip install -e .
```

## Quickstart

### 1) Define a config-carrying model

`ConfModule` is an Equinox module base class that requires a `conf` field.

```python
import equinox as eqx
import jax.random as jr
from jax import Array
from omegaconf import DictConfig, OmegaConf

from gearax.modules import ConfModule


class MyModel(ConfModule):
    # ConfModule already declares: conf: DictConfig = eqx.field(static=True)

    def __init__(self, conf: DictConfig, key: Array):
        self.conf = conf
        # initialize parameters using `key` and values in `conf`


conf = OmegaConf.create({"hidden_size": 64})
model = MyModel(conf, jr.key(0))
```

### 2) Save / load (config + parameters)

```python
from gearax.modules import save_model, load_model

save_model("model.zip", model)
restored = load_model("model.zip", MyModel)
assert isinstance(restored, MyModel)
```

Serialization format:
- `conf`: YAML (from `OmegaConf.to_yaml`)
- `pytree`: Equinox leaves (via `eqx.tree_serialise_leaves`)

### 3) Train with early stopping + progress

`gearax.trainer.train(...)` is a JAX/Equinox-focused training loop that expects:
- a `dataloader(train_set, batch_size, max_epoch, key)` yielding `(batch, epoch, batch_in_epoch)`
- `data_sharding` and `model_sharding` suitable for `eqx.filter_shard`

See `src/gearax/trainer.py` for the full signature and sharding/donation behavior.

## API Surface

Gearax does **not** currently re-export symbols from `gearax` top-level (it only defines `gearax.__version__`).
Import from submodules:

- `gearax.modules`
  - `ConfModule`
  - `save_model(path, model)`
  - `load_model(path, klass)`
- `gearax.trainer`
  - `Monitor` (early stopping + Rich progress)
  - `train(...)` (sharded training loop)
- `gearax.mixin`
  - `SubclassRegistryMixin` (+ `get_subclass(name)`)

## Development

```bash
# Tests
uv run pytest

# Lint/format (via pre-commit)
pre-commit run -a

# Build sdist/wheel
uv build

# Lockfile check (CI-friendly)
uv lock --check
```
