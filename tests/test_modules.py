import pytest


def test_confmodule():
    from gearax.modules import ConfModule
    from jax import Array
    import jax.random as jr
    import equinox as eqx
    from omegaconf import DictConfig

    class Derived(ConfModule):
        def __init__(self, conf: DictConfig, key: Array):
            self.conf = conf

    Derived(DictConfig({}), jr.key(0))

    with pytest.raises(TypeError):
        # ConfModule should not be instantiated
        ConfModule(DictConfig({}), jr.key(0))  # pyright: ignore[reportAbstractUsage]
