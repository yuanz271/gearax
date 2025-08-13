"""Gearax: A JAX-based machine learning utilities package.

Gearax provides utilities for training neural networks with JAX and Equinox,
including configuration management, model serialization, progress tracking,
and helpful mixins for ML workflows.

The package focuses on:
- Model configuration and serialization
- Training progress
- Data splitting utilities
- Subclass registry patterns

Examples
--------
>>> from gearax.helper import training_progress, arrays_split
>>> from gearax.modules import ConfModule, save_model, load_model
>>> from gearax.mixin import SubclassRegistryMixin
"""

__version__ = "0.1.0"
