from jax import numpy as jnp

from jax.numpy import (
    log,
    eye,
    exp,
    sum,
    diag,
    sqrt,
    mean,
    array,
    power,
    trace,
    zeros,
    argmax,
    matmul,
    transpose,
    concatenate,
)

from jax.numpy.linalg import (
    inv,
    det,
    cholesky,
)


__all__ = [
    "log",
    "eye",
    "exp",
    "sum",
    "diag",
    "sqrt",
    "mean",
    "array",
    "power",
    "trace",
    "zeros",
    "argmax",
    "matmul",
    "transpose",
    "concatenate",

    "inv",
    "det",
    "cholesky",

    "logsumexp",
    "log_softmax",
    "logdet",
    "matmul3",
    "kron_diag",
]


def logsumexp(data, axis=-1):
    data_max = jnp.max(data, axis=axis, keepdims=True)
    data_exp = jnp.exp(data - data_max)
    data_sum = jnp.log(jnp.sum(data_exp, axis=axis, keepdims=True))
    data_logsumexp = data_sum + data_max
    return data_logsumexp


def log_softmax(data):
    return data - logsumexp(data)


def logdet(data):
    sign, abslogdet = jnp.linalg.slogdet(data)
    return sign * abslogdet


def matmul3(mat0, mat1, mat2):
    return matmul(matmul(mat0, mat1), mat2)


def kron_diag(data, n):
    data_expanded = jnp.kron(jnp.eye(n), data)
    return data_expanded
