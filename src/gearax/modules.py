from dataclasses import InitVar
from typing import Any
from pathlib import Path
from zipfile import ZipFile

import equinox as eqx
from omegaconf import DictConfig, OmegaConf


class ConfModule(eqx.Module):
    conf: DictConfig = eqx.field(static=True)
    key: InitVar[Any]


def save_model(path: str | Path, model: ConfModule):
    with ZipFile(path, "w") as archive:
        archive.writestr("conf", OmegaConf.to_yaml(model.conf))
        with archive.open("pytree", "w") as f:
            eqx.tree_serialise_leaves(f, model)  # type: ignore


def load_model(path: str | Path, klass: type) -> ConfModule:
    with ZipFile(path) as archive:
        conf = OmegaConf.create(archive.read("conf").decode())
        model: ConfModule = eqx.filter_eval_shape(klass, OmegaConf.create(conf), None)
        with archive.open("pytree", "r") as f:
            model = eqx.tree_deserialise_leaves(f, model)  # type: ignore
    return model
