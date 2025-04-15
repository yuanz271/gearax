from pathlib import Path
from typing import Self, Type
from zipfile import ZipFile
import equinox as eqx
from omegaconf import DictConfig, OmegaConf


class Model(eqx.Module):
    conf: DictConfig
    
    @classmethod
    def load(cls, path: str | Path):
        return load_model(path, cls)
    
    @classmethod
    def save(cls, model: Self, path: str | Path):
        save_model(path, model)


def save_model(path, model: Model):
    with ZipFile(path, 'w') as archive:
        archive.writestr("conf", OmegaConf.to_yaml(model.conf))
        with archive.open("pytree", "w") as f:
            eqx.tree_serialise_leaves(f, model)


def load_model(path, klass: Type[Model]) -> Model:
    with ZipFile(path) as archive:
        conf = OmegaConf.create(archive.read("conf").decode())
        model = eqx.filter_eval_shape(klass, OmegaConf.create(conf))
        with archive.open("pytree", "r") as f:
            model = eqx.tree_deserialise_leaves(f, model)
    return model
