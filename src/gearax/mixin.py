class SubclassRegistryMixin:
    """
    A mixin class that provides a registry for subclasses.
    """

    _subclasses = dict()

    def __init_subclass__(cls, *args, **kwargs):
        super(cls).__init_subclass__(*args, **kwargs)
        if SubclassRegistryMixin in cls.__bases__:
            # Do not register immediate subclasses of this mixin
            return
        cls._subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, name: str) -> type:
        if name not in cls._subclasses:
            raise ValueError(f"Subclass {name} not found.")
        return cls._subclasses[name]
