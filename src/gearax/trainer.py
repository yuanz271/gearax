"""Training utilities for machine learning models.

This module provides training functions for neural networks using JAX and Equinox,
with support for efficient batch processing and gradient-based optimization.
"""

from collections.abc import Callable

from jax import Array, lax, numpy as jnp, random as jrnd
import optax
import equinox as eqx


@eqx.filter_jit
def train_epoch(
    model: eqx.Module,
    train_set: tuple[Array, ...],
    batch_loss_fn: Callable[[eqx.Module, tuple[Array, ...], Array], Array],
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    batch_size: int,
    key: Array,
) -> tuple[eqx.Module, optax.OptState]:
    """
    Perform gradient steps for a complete training epoch.

    This function executes a full epoch of training by iterating through
    batches of the training dataset, performing gradient descent steps,
    and computing validation loss. The training uses permuted batch sampling
    and JAX's while loop for efficient computation.

    Parameters
    ----------
    model : equinox.Module
        The XFADS model to be trained.
    train_set : tuple of jax.numpy.ndarray
        Training dataset as a tuple containing (times, observations, controls, covariates).
    batch_loss_fn : callable
        Function to compute batch loss. Should accept (model, batch, key) and return scalar loss.
    optimizer : optax.GradientTransformation
        Optax optimizer instance for gradient updates.
    opt_state : optax.OptState
        Current optimizer state.
    key : jax.random.PRNGKey
        Random key for stochastic operations.
    batch_size : int
        Number of samples per training batch.

    Returns
    -------
    equinox.Module
        Updated model after training epoch.

    Notes
    -----
    The function performs the following steps:
    1. Randomly permutes the training data for batch sampling
    2. Iterates through batches using JAX's while_loop for efficiency
    3. Applies gradient updates using the provided optimizer
    4. Computes validation loss in inference mode (no gradients)

    The training loop continues until all samples in the epoch have been processed.
    """
    train_size = jnp.size(train_set[0], 0)

    def batch_grad_step(model, opt_state, batch, key):
        """Perform one gradient update step."""
        vals, grads = eqx.filter_value_and_grad(batch_loss_fn)(
            model, batch, key
        )  # The gradient will be computed with respect to all floating-point JAX/NumPy arrays in the first argument
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        return vals, model, opt_state

    # batch loop
    params, static = eqx.partition(model, eqx.is_inexact_array)
    key, perm_key = jrnd.split(key)
    perm = jrnd.permutation(perm_key, train_size)

    def cond(carry):
        params, opt_state, batch_start, key = carry
        return batch_start + batch_size < train_size

    def batch_step(carry):
        """Single training step with validation."""
        params, opt_state, batch_start, key = carry
        model = eqx.combine(params, static)
        batch_idx = lax.dynamic_slice_in_dim(perm, batch_start, batch_size)
        batch = tuple(arr[batch_idx] for arr in train_set)
        key, grad_key = jrnd.split(key)
        loss, model, opt_state = batch_grad_step(model, opt_state, batch, grad_key)

        return (
            params,
            opt_state,
            batch_start + batch_size,
            key,
        )

    key, batch_key = jrnd.split(key)
    params, opt_state, *_ = lax.while_loop(
        cond,
        batch_step,
        (params, opt_state, 0, batch_key),
    )
    model = eqx.combine(params, static)

    return model, opt_state
