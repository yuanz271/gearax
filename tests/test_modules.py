from __future__ import annotations

from zipfile import ZipFile

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from omegaconf import DictConfig, OmegaConf


def test_confmodule():
    from gearax.modules import ConfModule
    from jax import Array

    class Derived(ConfModule):
        def __init__(self, conf: DictConfig, key: Array):
            self.conf = conf

    Derived(DictConfig({}), jr.key(0))

    with pytest.raises(TypeError):
        # ConfModule should not be instantiated
        ConfModule(DictConfig({}), jr.key(0))  # pyright: ignore[reportAbstractUsage]


def test_save_load_model_roundtrip(tmp_path):
    import equinox as eqx

    from gearax.modules import ConfModule, load_model, save_model

    class TinyModel(ConfModule):
        w: jax.Array

        def __init__(self, conf: DictConfig, key: jax.Array | None):
            self.conf = conf
            # load_model() uses eqx.filter_eval_shape(..., None), so tolerate key=None.
            if key is None:
                self.w = jnp.zeros((3,))
            else:
                self.w = jr.normal(key, (3,))

    path = tmp_path / "model.zip"
    conf = OmegaConf.create({"hidden_size": 3, "tag": "roundtrip"})

    model = TinyModel(conf, jr.key(0))
    save_model(path, model)

    restored = load_model(path, TinyModel)
    assert isinstance(restored, TinyModel)
    assert restored.conf["hidden_size"] == 3
    assert restored.conf["tag"] == "roundtrip"
    assert jnp.allclose(restored.w, model.w)

    # Ensure it remains a valid Equinox module.
    assert isinstance(restored, eqx.Module)


def test_save_model_writes_expected_zip_entries(tmp_path):
    from gearax.modules import ConfModule, save_model

    class TinyModel(ConfModule):
        w: jax.Array

        def __init__(self, conf: DictConfig, key: jax.Array | None):
            self.conf = conf
            self.w = jnp.ones((1,))

    path = tmp_path / "model.zip"
    save_model(path, TinyModel(OmegaConf.create({"x": 1}), None))

    with ZipFile(path) as zf:
        names = set(zf.namelist())
    assert names == {"conf", "pytree"}


@pytest.mark.parametrize("missing", ["conf", "pytree"])
def test_load_model_missing_zip_entry_raises(tmp_path, missing: str):
    from gearax.modules import ConfModule, load_model

    class TinyModel(ConfModule):
        w: jax.Array

        def __init__(self, conf: DictConfig, key: jax.Array | None):
            self.conf = conf
            self.w = jnp.zeros((1,))

    path = tmp_path / "broken.zip"
    with ZipFile(path, "w") as zf:
        if missing != "conf":
            zf.writestr("conf", "x: 1\n")
        if missing != "pytree":
            zf.writestr("pytree", b"not-a-pytree")

    with pytest.raises(KeyError):
        load_model(path, TinyModel)
