"""Mixins for common design patterns.

This module provides mixin classes that implement useful design patterns
for object-oriented programming, including automatic subclass registration.
"""

from typing import Any


class SubclassRegistryMixin:
    """Mixin that automatically registers subclasses for factory pattern.

    This mixin provides automatic registration of direct subclasses, allowing them
    to be retrieved by name. Each class that inherits from this mixin maintains
    its own separate registry containing only its immediate subclasses.

    The registry is populated automatically when subclasses are defined,
    using the `__init_subclass__` hook. Direct subclasses can then be retrieved
    by name using the `get_subclass` class method.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        """Hook called when a class is subclassed.

        Automatically registers the subclass with its immediate parent class.
        Each class gets its own registry and only tracks its direct subclasses,
        not descendants further down the hierarchy.

        Parameters
        ----------
        cls : type
            The class being defined as a subclass.
        **kwargs : dict
            Additional keyword arguments passed to super().__init_subclass__.
        """
        super().__init_subclass__(**kwargs)

        # This ensures each inheritance chain gets its own registry
        if not hasattr(
            cls, "_subclasses"
        ):
            # Initialize the registry for the root class of this inheritance chain
            # This class will collect all its direct subclasses
            cls._subclasses = {}
        # This is a subclass of an already-registered class
        # Register it with its immediate parent's registry
        cls._subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, name: str) -> type[Any]:
        """Retrieve a registered subclass by name.

        Parameters
        ----------
        name : str
            The name of the subclass to retrieve (class.__name__).

        Returns
        -------
        type
            The subclass type with the given name.

        Raises
        ------
        ValueError
            If no subclass with the given name is found in the registry.

        Examples
        --------
        >>> class Base(SubclassRegistryMixin):
        ...     pass
        >>> class Child(Base):
        ...     pass
        >>> Child is Base.get_subclass("Child")
        True
        """
        if name not in cls._subclasses:
            raise ValueError(f"Subclass {name} not found.")
        return cls._subclasses[name]
