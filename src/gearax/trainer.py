"""Training utilities for machine learning models.

This module provides training functions for neural networks using JAX and Equinox,
with support for efficient batch processing and gradient-based optimization.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import optax
from jax import Array, lax
from jax import numpy as jnp
from jax import random as jr


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
    key, perm_key = jr.split(key)
    perm = jr.permutation(perm_key, train_size)

    def cond(carry):
        params, opt_state, batch_start, key = carry
        return batch_start + batch_size < train_size

    def batch_step(carry):
        """Single training step with validation."""
        params, opt_state, batch_start, key = carry
        model = eqx.combine(params, static)
        batch_idx = lax.dynamic_slice_in_dim(perm, batch_start, batch_size)
        batch = tuple(arr[batch_idx] for arr in train_set)
        key, grad_key = jr.split(key)
        loss, model, opt_state = batch_grad_step(model, opt_state, batch, grad_key)

        return (
            params,
            opt_state,
            batch_start + batch_size,
            key,
        )

    key, batch_key = jr.split(key)
    params, opt_state, *_ = lax.while_loop(
        cond,
        batch_step,
        (params, opt_state, 0, batch_key),
    )
    model = eqx.combine(params, static)

    return model, opt_state


def train(model, train_set, valid_set, key, batch_loss_fun, dataloader, batch_size, num_epochs, patience, optimizer, data_sharding, model_sharding, epoch_callback=None):
    @eqx.filter_jit(donate="all")
    def train_step(model, opt_state, batch, key, data_sharding, model_sharding):
        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)

        grads = eqx.filter_grad(batch_loss_fun)(model, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

        return model, opt_state

    @eqx.filter_jit(donate="all-except-first")
    def evaluate(model, batch, key, data_sharding, model_sharding):
        model = eqx.filter_shard(eqx.nn.inference_mode(model), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)
        return lax.stop_gradient(batch_loss_fun(model, batch, key))

    # num_devices = len(jax.devices())
    # mesh = jax.make_mesh((num_devices,), ("batch",))
    # data_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # model_sharding = NamedSharding(mesh, PartitionSpec())

    # devices_shape = data_sharding.mesh.devices.shape
    # print(devices_shape)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    key, loader_key = jr.split(key)  # Key for dataloader

    # Best model tracking variables
    best_model = model
    key, valid_key = jr.split(key)
    # valid_key = jr.split(valid_key, devices_shape)
    best_val_loss = evaluate(best_model, valid_set, valid_key, data_sharding, model_sharding)
    patience_left = patience

    # Training loop with per-epoch validation and best model tracking
    for batch, epoch, batch_in_epoch in dataloader(train_set, batch_size, num_epochs, loader_key):
        try:
            key, batch_key = jr.split(key)
            # batch_key = jr.split(batch_key, devices_shape)
            batch = eqx.filter_shard(batch, data_sharding) #?
            model, opt_state = train_step(model, opt_state, batch, batch_key, data_sharding, model_sharding)

            # Evaluate at the start of each new epoch
            if batch_in_epoch == 0:
                # Evaluate on validation set only
                key, valid_key = jr.split(key)
                # valid_key = jr.split(valid_key, devices_shape)
                val_loss = evaluate(model, valid_set, valid_key, data_sharding, model_sharding)

                if epoch_callback is not None:
                    epoch_callback(epoch, val_loss)

                # Best model tracking
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Create independent copy using JAX tree utilities with array filtering
                    best_model = jax.tree.map(
                        lambda x: jnp.copy(x) if eqx.is_array(x) else x,
                        model
                    )
                    patience_left = patience
                else:
                    patience_left -= 1

                # Optional: Early stopping
                if patience_left == 0:
                    break
        except KeyboardInterrupt:
            # print(f"\nTraining interrupted at epoch {epoch}! Using best model found so far...")
            break
    else:
        # Final validation check
        key, valid_key = jr.split(key)
        # valid_key = jr.split(valid_key, devices_shape)
        val_loss = evaluate(model, valid_set, valid_key, data_sharding, model_sharding)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create independent copy using JAX tree utilities with array filtering
            best_model = jax.tree.map(
                lambda x: jnp.copy(x) if eqx.is_array(x) else x,
                model
            )

    return best_model
