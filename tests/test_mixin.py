import pytest
from gearax.mixin import SubclassRegistryMixin


# Helper classes for testing
class Base(SubclassRegistryMixin):
    pass


class ChildA(Base):
    pass


class ChildB(Base):
    pass


def test_get_subclass_success():
    assert Base.get_subclass("ChildA") is ChildA
    assert Base.get_subclass("ChildB") is ChildB


def test_get_subclass_failure():
    with pytest.raises(ValueError, match="Subclass NonExistent not found."):
        Base.get_subclass("NonExistent")


def test_registry_isolation():
    class AnotherBase(SubclassRegistryMixin):
        pass

    class AnotherChild(AnotherBase):
        pass

    # ChildA and ChildB should not be in AnotherBase's registry
    with pytest.raises(ValueError):
        AnotherBase.get_subclass("ChildA")
    with pytest.raises(ValueError):
        AnotherBase.get_subclass("ChildB")
    # AnotherChild should be in AnotherBase's registry
    assert AnotherBase.get_subclass("AnotherChild") is AnotherChild
