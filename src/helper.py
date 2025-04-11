import math
import time

from rich.progress import (
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
)


def training_progress():
    return Progress(
        SpinnerColumn(),  # Include default columns
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        TextColumn("•"),
        "Elapsed",
        TimeElapsedColumn(),
        TextColumn("•"),
        "Remainning",
        TimeRemainingColumn(),
        TextColumn("•"),
        "Loss",
        TextColumn("{task.fields[loss]:.3f}")
    )


def arrays_split(arrays, *, rng, ratio=None, size=None):
    dataset_size = arrays[0].shape[0]
    if size is None:
        size = int(ratio * dataset_size)
    perm = rng.permutation(dataset_size)

    return tuple(array[perm[size:]] for array in arrays), tuple(
        array[perm[:size]] for array in arrays
    )


if __name__ == "__main__":
    with training_progress() as pbar:
        task_id = pbar.add_task("Fitting", total=100, loss=math.inf)
        for i in range(100):
            time.sleep(0.1)
            pbar.update(task_id, advance=1, loss=i/100)
