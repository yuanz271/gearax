from typing import Generator
from numpy import random as nprnd


def dataloader(
    arrays: tuple, *, batch_size: int, rng: nprnd.Generator
) -> Generator[tuple[int, int, tuple]]:
    dataset_size = arrays[0].shape[0]
    epoch_idx = 0
    while True:
        perm = rng.permutation(dataset_size)
        start = 0
        end = batch_size
        batch_idx = 0
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield epoch_idx, batch_idx, tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
            batch_idx += 0
        epoch_idx += 1
