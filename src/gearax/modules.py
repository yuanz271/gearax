from jax import lax, numpy as jnp
import equinox as eqx


class ConcatInput(eqx.Module):
    """Merge
    """
    network: eqx.Module
    axis: int = eqx.field(default=0, static=True)

    def __call__(self, *args, **kwargs):
        return self.network(jnp.concatenate(args, axis=self.axis), **kwargs)
