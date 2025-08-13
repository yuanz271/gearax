# Gearax

A JAX-based machine learning utilities package that provides essential tools for training neural networks with JAX and Equinox. Gearax focuses on configuration management, model serialization, progress tracking, and helpful design patterns for ML workflows.

## Features

- **Model Configuration & Serialization**: Seamlessly combine Equinox models with OmegaConf configurations and save/load complete models
- **Training Progress Tracking**: Rich progress bars optimized for machine learning training loops
- **Data Splitting Utilities**: Convenient functions for splitting datasets into training/validation sets
- **Design Pattern Mixins**: Automatic subclass registration for factory patterns

## Installation

```bash
pip install git+https://github.com/yourusername/gearax.git
```

## API Reference

### `gearax.modules`

#### `ConfModule`
Base class for Equinox modules with configuration management.

**Attributes:**
- `conf`: DictConfig containing hyperparameters and settings
- `key`: InitVar for random key (not stored as instance attribute)

#### `save_model(path, model)`
Save a ConfModule to a ZIP archive.

**Parameters:**
- `path`: File path for the saved model
- `model`: ConfModule instance to save

#### `load_model(path, klass)`
Load a ConfModule from a ZIP archive.

**Parameters:**
- `path`: File path to the saved model
- `klass`: Model class to instantiate

**Returns:** Loaded model instance

### `gearax.helper`

#### `training_progress()`
Create a Rich Progress bar configured for training loops.

**Returns:** Configured Progress instance with training-specific columns

#### `arrays_split(arrays, *, rng, ratio=None, size=None)`
Split arrays into training and validation sets.

**Parameters:**
- `arrays`: Sequence of arrays to split
- `rng`: Random number generator
- `ratio`: Fraction for validation set (0 < ratio < 1)
- `size`: Absolute number of validation samples

**Returns:** Tuple of (train_arrays, val_arrays)

### `gearax.mixin`

#### `SubclassRegistryMixin`
Mixin for automatic subclass registration.

**Methods:**
- `get_subclass(name)`: Retrieve registered subclass by name
