from equinox import Module


class SubClassRegistry(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)  # class initialization (not the metaclass)
        if not hasattr(cls, "registry"):
            # This is the base class: create an empty registry
            cls.registry = {}
        else:
            # This is a subclass: register it
            cls.registry[cls.__name__] = cls

        def get_subclass(cls, name: str) -> type:
            if name not in cls.registry:
                raise ValueError(f"{name} is not a subclass of {cls.__name__}.")
            return cls.registry[name]

        cls.get_subclass = classmethod(get_subclass)


class ModuleRegistry(SubClassRegistry, type(Module)): ...
