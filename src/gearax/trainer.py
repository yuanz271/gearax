"""Training utilities for machine learning models.

This module provides training functions for neural networks using JAX and Equinox,
with support for efficient batch processing and gradient-based optimization.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import equinox as eqx
import jax
from jax import Array, lax
from jax import numpy as jnp
from jax import random as jr
from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def training_progress():
    """
    Create a Rich progress bar for training visualization.

    Returns
    -------
    Progress
        Configured Rich Progress instance with columns for:
        - Spinner animation
        - Task description
        - Progress counter
        - Elapsed time
        - Remaining time estimate
        - Current loss value
        - Moving average loss

    Notes
    -----
    The progress bar provides real-time feedback during training including
    current and smoothed loss values to monitor convergence.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        "Epoch",
        MofNCompleteColumn(),
        # TextColumn("•"),
        "Elapsed",
        TimeElapsedColumn(),
        "Remaining",
        TimeRemainingColumn(),
        "Loss",
        TextColumn("{task.fields[loss]:.3f}"),
        "Best",
        TextColumn("{task.fields[best]:.3f}"),
    )


def _copy_pytree(pt):
    return jax.tree.map(lambda x: jnp.copy(x) if eqx.is_array(x) else x, pt)


@dataclass
class Monitor:
    evaluate: Callable
    valid_set: Any
    patience: int
    best_model: eqx.Module
    best_loss: float
    callback: Callable | None = None
    patience_left: int = field(init=False)
    losses: list = field(init=False, default_factory=list)
    _pbar: Any = field(init=False)

    def __init__(self, model, valid_set, eval_fun, max_epoch, patience):
        self.evaluate = eval_fun
        self.valid_set = valid_set
        self.patience = patience
        self.patience_left = patience

        self.best_model = _copy_pytree(model)
        self.best_loss = jnp.inf
        self.losses = []

        self._pbar = training_progress()
        self._task_id = self._pbar.add_task(
            "Training", total=max_epoch, loss=jnp.inf, best=jnp.inf
        )
        self._pbar.start()

    def step(self, model, key: Array):
        val_loss = self.evaluate(model, self.valid_set, key).item()
        self.losses.append(val_loss)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = _copy_pytree(model)
            self.patience_left = self.patience
        else:
            self.patience_left -= 1

        jax.debug.callback(
            lambda vl, bl: self._pbar.update(
                self._task_id, advance=1, loss=vl, best=bl
            ),
            val_loss,
            self.best_loss,
        )  # type: ignore

        return self.patience_left > 0

    def stop(self):
        self._pbar.stop()


def train(
    model,
    train_set,
    valid_set,
    key,
    batch_loss_fun,
    dataloader,
    batch_size,
    max_epoch,
    patience,
    optimizer,
    data_sharding,
    model_sharding,
):
    @eqx.filter_jit(donate="all")
    def train_step(model, opt_state, batch, key, data_sharding, model_sharding):
        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)

        grads = eqx.filter_grad(batch_loss_fun)(model, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)

        model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

        return model, opt_state

    @eqx.filter_jit
    def evaluate(model, batch, key, data_sharding, model_sharding):
        model = eqx.filter_shard(eqx.nn.inference_mode(model), model_sharding)
        batch = eqx.filter_shard(batch, data_sharding)
        return lax.stop_gradient(batch_loss_fun(model, batch, key))

    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    monitor = Monitor(
        model,
        valid_set,
        partial(evaluate, data_sharding=data_sharding, model_sharding=model_sharding),
        max_epoch,
        patience,
    )

    # Training loop with per-epoch validation and best model tracking
    key, loader_key = jr.split(key)  # Key for dataloader
    for batch, epoch, batch_in_epoch in dataloader(
        train_set, batch_size, max_epoch, loader_key
    ):
        try:
            key, batch_key = jr.split(key)
            model, opt_state = train_step(
                model, opt_state, batch, batch_key, data_sharding, model_sharding
            )

            # Evaluate at the start of each new epoch
            if batch_in_epoch == 0:
                # Evaluate on validation set only
                key, monitor_key = jr.split(key)
                if not monitor.step(model, monitor_key):
                    break

        except KeyboardInterrupt:
            break
    else:
        # Final validation check
        key, monitor_key = jr.split(key)
        monitor.step(model, monitor_key)

    monitor.stop()

    return monitor.best_model
