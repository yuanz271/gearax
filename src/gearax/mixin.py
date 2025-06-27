class SubclassRegistryMixin:
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        # Only set up registry for the first subclass of the mixin
        if cls in SubclassRegistryMixin.__subclasses__() and not hasattr(
            cls, "_subclasses"
        ):
            # Initialize the registry if it doesn't exist
            cls._subclasses = dict()
        else:
            cls._subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, name: str) -> type:
        if name not in cls._subclasses:
            raise ValueError(f"Subclass {name} not found.")
        return cls._subclasses[name]
