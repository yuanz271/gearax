from typing import Any


def with_subclass_registry(cls) -> Any:  # Any for type checker compatibility
    """Decorator to register a class in the ClassRegistry."""

    if not hasattr(cls, "registry"):
        cls.registry = dict()

    def __init_subclass__(subcls, *args, **kwargs):
        super(subcls).__init_subclass__(*args, **kwargs)
        cls.registry[subcls.__name__] = subcls

    def get_subclass(cls, name: str) -> type:
        if name not in cls.registry:
            raise ValueError(f"Subclass {name} not found.")
        return cls.registry[name]

    cls.__init_subclass__ = classmethod(__init_subclass__)
    cls.get_subclass = classmethod(get_subclass)

    return cls
