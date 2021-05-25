from jax import numpy as jnp
from jax import random
import numpy as np


__all__ = [
    "TestBatch",
    "TrainBatch",
    "logsumexp",
    "log_softmax",
    "robust_max_values",
    "kron_diag",
    "matmul2",
    "matmul3",
    "split_kernel",
    "get_true_values",
    "permute_dataset",
]


class TrainBatch:
    def __init__(self, x, y, batch_size, epochs=100, seed=0):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

    def __iter__(self):
        self.key = random.PRNGKey(self.seed)
        self.epoch = 0
        return self

    def __len__(self):
        return self.epochs

    def __next__(self):
        if self.epoch >= self.epochs:
            raise StopIteration
        else:
            self.epoch += 1

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
        self.batch_len = (x.shape[0] // batch_size) + (0 if x.shape[0] % batch_size else 1)

    def __iter__(self):
        self.batch_i = 0
        return self

    def __len__(self):
        return self.batch_len

    def __next__(self):
        if self.batch_i >= self.batch_len:
            raise StopIteration
        else:
            self.batch_i += 1

        batch_start = self.batch_i * self.batch_size
        batch_end = batch_start + self.batch_size
        x_batch = self.x[batch_start: batch_end]
        y_batch = self.y[batch_start: batch_end]

        return x_batch, y_batch


def permute_dataset(data, label, seed=0):
    idx = np.random.RandomState(seed).permutation(data.shape[0])
    permuted_data = data[idx]
    permuted_label = label[idx]
    return permuted_data, permuted_label


def logsumexp(data):
    data_max = jnp.max(data, axis=-1, keepdims=True)
    data_exp = jnp.exp(data - data_max)
    data_sum = jnp.log(jnp.sum(data_exp, axis=-1, keepdims=True))
    data_logsumexp = data_sum + data_max
    return data_logsumexp


def log_softmax(data):
    data_log_softmax = data - logsumexp(data)
    return data_log_softmax


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


def kron_diag(data, n):
    data_expanded = jnp.kron(jnp.eye(n), data)
    return data_expanded


def matmul2(mat0, mat1):
    mul = jnp.matmul(mat0, mat1)
    return mul


def matmul3(mat0, mat1, mat2):
    mul = jnp.matmul(jnp.matmul(mat0, mat1), mat2)
    return mul


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
