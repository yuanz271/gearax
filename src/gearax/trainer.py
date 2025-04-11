import math
import numpy as np
from jax import vmap, numpy as jnp
import equinox as eqx
from data import dataloader
import optax
import rich

import helper


class Trainer:
    def __init__(self, learning_rate=1e-2, max_iter=500):
        self._progress =  helper.training_progress()

        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.best_model = None
        self.best_loss = None

    def record(self, model, loss: float) -> bool:
        should_stop = False
        self._losses.append(loss)

        if loss < self.min_loss:
            self.min_loss = loss
            self.best_model = model

        return should_stop
    
    def on_training_begin(self, model, loss):
        self.best_model = model 
        self.best_loss = loss
        self.task_id = self._progress.add_task("Fitting", total=self.max_iter, loss=loss)

    def on_training_iter(self, model, loss):
        self._progress.update(self.task_id, advance=1, loss=loss)

        if loss < self.best_loss:
            self.best_model = model 
            self.best_loss = loss

    def on_training_end(self):
        pass

    def fit(self, model: eqx.Module, data, *, seed, loss_func, make_optimizer, validation_ratio=None, validation_size=None) -> eqx.Module:
        rng = np.random.default_rng(seed)
        train_set, valid_set = helper.arrays_split(
            data, rng=rng, ratio=validation_ratio, size=validation_size
        )

        optimizer = make_optimizer(self.learning_rate)

        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def batch_loss(model, batch):
            inputs, targets = batch
            outputs = vmap(model)(inputs)
            losses = vmap(loss_func)(outputs, targets)
            return jnp.mean(losses)
            
        @eqx.filter_jit
        def step(model, batch, opt_state):
            lss, grads = eqx.filter_value_and_grad(batch_loss)(model, batch)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)

            return lss, model, opt_state
        
        valid_loss = batch_loss(model, *valid_set).item()
        self.on_training_begin(model, valid_loss)

        terminate = False
        for i, batch in enumerate(dataloader(train_set, batch_size=self.batch_size, rng=rng)):
            try:
                if terminate:
                    break
                _, model, opt_state = step(model, batch, opt_state)
                lss = batch_loss(model, *valid_set).item()
                terminate = self.on_training_iter(model, lss)
            except KeyboardInterrupt:
                terminate = True
        
        self.on_training_end()

        return self.best_model
