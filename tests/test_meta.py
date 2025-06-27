import pytest
from gearax.modules import ConfModule
from gearax.meta import ModuleRegistry


class BaseA(ConfModule, metaclass=ModuleRegistry):
    pass


class BaseB(ConfModule, metaclass=ModuleRegistry):
    pass


class SubA(BaseA):
    pass


class SubB(BaseB):
    pass


def test_registry_contains_subclasses():
    assert "SubA" in BaseA.registry
    assert "SubB" in BaseB.registry


def test_get_subclass_success():
    assert BaseA.get_subclass("SubA") is SubA
    assert BaseB.get_subclass("SubB") is SubB


def test_registry_isolation():
    with pytest.raises(ValueError) as e:
        BaseA.get_subclass("SubB")
    assert "SubB is not a subclass of BaseA." in str(e.value)

    with pytest.raises(ValueError) as e:
        BaseB.get_subclass("SubA")
    assert "SubA is not a subclass of BaseB." in str(e.value)
