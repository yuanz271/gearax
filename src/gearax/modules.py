"""Model configuration and serialization utilities.

This module provides utilities for creating, saving, and loading machine learning
models that combine Equinox neural networks with OmegaConf configuration management.
Models are serialized to ZIP archives containing both the configuration and
the model parameters.

The abstract/final design pattern

Every subclass of eqx.Module must be either

(a) abstract (it can be subclassed, but not instantiated); or
(b) final (it can be instantiated, but not subclassed).

Only abstractmethods and abstract attributes can be overridden. (So once they've been implemented, then a subclass may not override them.)

The __init__ method, and all dataclass fields, must all be defined in one class. (No defining fields in multiple parts of a class hierarchy.)
"""

from abc import abstractmethod
from pathlib import Path
from zipfile import ZipFile

import equinox as eqx
from jax import Array
from omegaconf import DictConfig, OmegaConf


class ConfModule(eqx.Module):
    """Base class for Equinox modules with configuration management.

    This class combines an Equinox Module with an OmegaConf DictConfig,
    allowing models to carry their configuration alongside their parameters.

    Attributes
    ----------
    conf : DictConfig
        The configuration dictionary containing hyperparameters and settings.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> conf = OmegaConf.create({"hidden_size": 64, "num_layers": 2})
    >>> class MyModel(ConfModule):
    ...     conf: DictConfig = eqx.field(static=True)
    ...
    ...     def __init__(self, conf: DictConfig, key: Array):
    ...         self.conf = conf
    ...         # Initialize layers based on conf...
    """

    conf: DictConfig = eqx.field(static=True)

    @abstractmethod
    def __init__(self, conf: DictConfig, key: Array) -> None: ...


def save_model(path: str | Path, model: ConfModule) -> None:
    """Save a ConfModule to a ZIP archive.

    Serializes both the model configuration and parameters to a ZIP file.

    Parameters
    ----------
    path : str or Path
        File path where the model should be saved. Will be created or
        overwritten if it exists.
    model : ConfModule
        The model instance to save, containing both configuration and
        trained parameters.

    Examples
    --------
    >>> save_model("model.zip", trained_model)
    """
    with ZipFile(path, "w") as archive:
        archive.writestr("conf", OmegaConf.to_yaml(model.conf))
        with archive.open("pytree", "w") as f:
            eqx.tree_serialise_leaves(f, model)  # type: ignore


def load_model(path: str | Path, klass: type[ConfModule]) -> ConfModule:
    """Load a ConfModule from a ZIP archive.

    Deserializes a model that was previously saved with save_model().

    Parameters
    ----------
    path : str or Path
        File path to the saved model ZIP archive.
    klass : type[ConfModule]
        The model class to instantiate. Must be a subclass of ConfModule
        and have a constructor that accepts (conf, key) arguments.

    Returns
    -------
    ConfModule
        The loaded model instance with configuration and trained parameters.
        Note: The actual return type will be an instance of the provided klass,
        but the static type is ConfModule. Use type casting if you need
        access to subclass-specific methods.

    Examples
    --------
    >>> loaded_model = load_model("model.zip", MyModel)
    >>> assert isinstance(loaded_model, MyModel)
    """
    with ZipFile(path) as archive:
        conf = OmegaConf.create(archive.read("conf").decode())
        model: ConfModule = eqx.filter_eval_shape(klass, OmegaConf.create(conf), None)
        with archive.open("pytree", "r") as f:
            model = eqx.tree_deserialise_leaves(f, model)  # type: ignore
        return model
