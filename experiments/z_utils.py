import numpy as np

from jax import random
from jax import numpy as jnp


__all__ = [
    "TestBatch",
    "TrainBatch",
    "robust_max_values",
    "split_kernel",
    "get_true_values",
]


class TrainBatch:
    def __init__(self, x, y, batch_size, steps=100, seed=0):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.steps = steps
        self.seed = seed

    def __iter__(self):
        self.key = random.PRNGKey(self.seed)
        self.step = 0
        return self

    def __len__(self):
        return self.steps

    def __next__(self):
        if self.step >= self.steps:
            raise StopIteration
        else:
            self.step += 1

        self.key, split = random.split(self.key)

        random_idxs = random.permutation(split, jnp.arange(self.x.shape[0], dtype=int))
        random_x = self.x[random_idxs]
        random_y = self.y[random_idxs]

        x_batch = random_x[:self.batch_size]
        y_batch = random_y[:self.batch_size]

        return x_batch, y_batch


class TestBatch:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.batch_len = (x.shape[0] // batch_size) + (1 if x.shape[0] % batch_size else 0)

    def __iter__(self):
        self.batch_i = 0
        return self

    def __len__(self):
        return self.batch_len

    def __next__(self):
        if self.batch_i >= self.batch_len:
            raise StopIteration

        batch_start = self.batch_i * self.batch_size
        batch_end = batch_start + self.batch_size
        x_batch = self.x[batch_start: batch_end]
        y_batch = self.y[batch_start: batch_end]

        self.batch_i += 1

        return x_batch, y_batch


# TODO
def robust_max_values(data, label, class_num, eps=0.001):
    data_idx = jnp.argmax(data, axis=-1)
    label_idx = jnp.argmax(label, axis=-1)

    out = jnp.sum((data_idx == label_idx) == True) * (1 - eps) \
        + jnp.sum((data_idx == label_idx) == False) * (eps / (class_num - 1))

    # out = jnp.full_like(data, 1 - eps)
    # idxs = jnp.argmax(data, axis=-1)[..., jnp.newaxis]
    # np.put_along_axis(out, idxs, 1 - eps, axis=-1)  # TODO: replace with jax
    return out


def split_kernel(kernel, num_11):
    kernel_11 = kernel[:num_11, :num_11]
    kernel_12 = kernel[:num_11, num_11:]
    kernel_21 = kernel[num_11:, :num_11]
    kernel_22 = kernel[num_11:, num_11:]
    return kernel_11, kernel_12, kernel_21, kernel_22


def get_true_values(value, label):
    """
    Parameters
    ----------
    value: [sample_num, batch_num, class_num]
    label: [batch_num, class_num]

    Returns
    -------
    true_values: [sample_num, batch_num]
    """

    sample_num = value.shape[0]

    label_idx = jnp.argmax(label, axis=-1)[jnp.newaxis, :, jnp.newaxis]
    value_idx = jnp.repeat(label_idx, sample_num, axis=0)

    true_values = jnp.take_along_axis(value, value_idx, axis=-1).squeeze(axis=-1)

    return true_values
