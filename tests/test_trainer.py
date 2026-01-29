from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr


def test_monitor_patience_and_min_epoch_behavior():
    from gearax.trainer import Monitor

    # A tiny model with one parameter.
    model = {"w": jnp.array([0.0])}

    # Sequence: improve once, then flat.
    # With min_epoch=3, the first no-improvement step should NOT decrement patience_left.
    losses = iter([10.0, 9.0, 9.0, 9.0, 9.0])

    def eval_fun(_model, _valid_set, _key):
        return jnp.array(next(losses))

    monitor = Monitor(
        model=model,
        valid_set=None,
        eval_fun=eval_fun,
        max_epoch=10,
        patience=2,
        min_epoch=3,
    )
    try:
        key = jr.key(0)

        # Step 1: improvement (best_loss inf -> 10)
        assert monitor.step(model, key) is True
        assert monitor.best_loss == 10.0
        assert monitor.patience_left == 2

        # Step 2: improvement (10 -> 9)
        assert monitor.step(model, key) is True
        assert monitor.best_loss == 9.0
        assert monitor.patience_left == 2

        # Step 3: no improvement, but still within min_epoch window => no decrement
        assert monitor.step(model, key) is True
        assert monitor.patience_left == 2

        # Step 4: no improvement, now decrement
        assert monitor.step(model, key) is True
        assert monitor.patience_left == 1

        # Step 5: no improvement, decrement to 0 => stop condition false
        assert monitor.step(model, key) is False
        assert monitor.patience_left == 0
    finally:
        monitor.stop()


@dataclass
class _TinySGD:
    lr: float = 0.1

    def init(self, _params):
        return None

    def update(self, grads, state, _params):
        updates = jax.tree.map(lambda g: -self.lr * g, grads)
        return updates, state


def test_train_smoke_single_device_sharding():
    import equinox as eqx

    from gearax.trainer import train

    class Model(eqx.Module):
        w: jax.Array

        def __init__(self, key):
            self.w = jr.normal(key, (1,))

        def __call__(self, x):
            return x * self.w

    model = Model(jr.key(0))
    train_set = None
    valid_set = jnp.array([1.0])

    def batch_loss_fun(m, batch, _key):
        pred = m(batch)
        return jnp.mean((pred - batch) ** 2)

    def dataloader(_train_set, _batch_size, max_epoch, _loader_key):
        # 1 batch per epoch.
        for epoch in range(max_epoch):
            yield jnp.array([1.0]), epoch, 0

    device_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    best_model = train(
        model=model,
        train_set=train_set,
        valid_set=valid_set,
        key=jr.key(123),
        batch_loss_fun=batch_loss_fun,
        dataloader=dataloader,
        batch_size=1,
        max_epoch=3,
        patience=1,
        optimizer=_TinySGD(lr=0.01),
        data_sharding=device_sharding,
        model_sharding=device_sharding,
        min_epoch=0,
    )

    assert isinstance(best_model, Model)
    assert best_model.w.shape == (1,)
