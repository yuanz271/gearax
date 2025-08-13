"""Training and utility helper functions.

This module provides utility functions for machine learning workflows,
including progress tracking for training loops and data splitting utilities.
"""

from typing import Sequence, Any
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)


def training_progress() -> Progress:
    """Create a rich Progress bar configured for training loops.

    Returns a Rich Progress instance with columns optimized for displaying
    training progress, including spinner, task description, completion count,
    elapsed time, remaining time, and current loss value.

    Returns
    -------
    Progress
        A configured Rich Progress instance with training-specific columns.

    Examples
    --------
    >>> progress = training_progress()
    >>> task_id = progress.add_task("Training", total=100)
    >>> progress.update(task_id, advance=1, loss=0.5)
    """
    return Progress(
        SpinnerColumn(),  # Include default columns
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TextColumn("•"),
        "Elapsed",
        TimeElapsedColumn(),
        TextColumn("•"),
        "Remaining",
        TimeRemainingColumn(),
        TextColumn("•"),
        "Loss",
        TextColumn("{task.fields[loss]:.3f}"),
    )


def arrays_split(
    arrays: Sequence[Any],
    *,
    rng: Any,
    ratio: float | None = None,
    size: int | None = None
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Split arrays into training and validation sets.

    Randomly splits a collection of arrays along their first axis into
    training and validation sets. All arrays must have the same size
    along the first dimension.

    Parameters
    ----------
    arrays : sequence of array-like
        Collection of arrays to split. All arrays must have the same
        first dimension size.
    rng : numpy.random.Generator
        Random number generator for shuffling.
    ratio : float, optional
        Fraction of data to use for validation set (0 < ratio < 1).
        Mutually exclusive with size parameter.
    size : int, optional
        Absolute number of samples for validation set.
        Mutually exclusive with ratio parameter.

    Returns
    -------
    tuple
        A tuple containing (train_arrays, val_arrays) where:
        - train_arrays: tuple of arrays for training
        - val_arrays: tuple of arrays for validation

    Raises
    ------
    ValueError
        If arrays is empty, if neither ratio nor size is provided,
        if both ratio and size are provided, if ratio is not between 0 and 1,
        if size is negative or larger than dataset size, or if arrays
        have mismatched first dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100,)
    >>> (X_train, y_train), (X_val, y_val) = arrays_split([X, y], rng=rng, ratio=0.2)
    >>> X_train.shape[0] + X_val.shape[0] == 100
    True
    """
    if not arrays:
        raise ValueError("arrays cannot be empty")

    dataset_size = arrays[0].shape[0]

    if size is None and ratio is None:
        raise ValueError("Either ratio or size must be provided")

    if size is None:
        if ratio is None or not (0 < ratio < 1):
            raise ValueError("ratio must be between 0 and 1")
        size = int(ratio * dataset_size)

    if size < 0 or size > dataset_size:
        raise ValueError(f"size must be between 0 and {dataset_size}")

    # Validate all arrays have the same first dimension
    for i, array in enumerate(arrays):
        if array.shape[0] != dataset_size:
            raise ValueError(f"Array {i} has shape {array.shape[0]} but expected {dataset_size}")

    perm = rng.permutation(dataset_size)

    return tuple(array[perm[size:]] for array in arrays), tuple(
        array[perm[:size]] for array in arrays
    )
