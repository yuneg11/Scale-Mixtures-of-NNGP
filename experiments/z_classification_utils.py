from jax import jit

from neural_tangents import stax
from neural_tangents.predict import gradient_descent_mse_ensemble

from .ops import *
from .utils import *

__all__ = [
    "get_cnn_kernel",
    "get_mlp_kernel",
    "log_likelihood",
    "test_log_likelihood",
    "get_correct_count",
    "mean_covariance",
]


def get_cnn_kernel(
    depth,
    class_num,
    act="relu",
    W_std=1.,
    b_std=0.,
    last_W_std=1.,
):
    if act == "relu":
        act_class = stax.Relu
    elif act == "erf":
        act_class = stax.Erf
    else:
        raise KeyError("Unsupported act '{}'".format(act))

    layers = []
    for _ in range(depth):
        layers.append(stax.Conv(1, (3, 3), (1, 1), 'SAME', W_std=W_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(class_num, W_std=last_W_std))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return  kernel_fn


def get_mlp_kernel(
    depth,
    class_num,
    act="relu",
    W_std=1.,
    b_std=0.,
    last_W_std=1.,
):
    if act == "relu":
        act_class = stax.Relu
    elif act == "erf":
        act_class = stax.Erf
    else:
        raise KeyError("Unsupported act '{}'".format(act))

    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(512, W_std=W_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Dense(class_num, W_std=last_W_std))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return  kernel_fn


def log_likelihood(label, sampled_f, train_num):
    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_values(sampled_f_softmax, label)
    ll = train_num * mean(mean(true_label_softmax, axis=1))
    return ll


def test_log_likelihood(label, sampled_f, sample_num):
    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_values(sampled_f_softmax, label)
    ll = sum(logsumexp(true_label_softmax.T) - log(sample_num))
    return ll


def get_correct_count(label, sampled_f):
    sampled_f_softmax = exp(log_softmax(sampled_f))
    mean_f_softmax = mean(sampled_f_softmax, axis=0)
    predict_y = argmax(mean_f_softmax, axis=-1)
    true_y = argmax(label, axis=-1)
    correct_count = sum(predict_y == true_y)
    return correct_count


def mean_covariance(
    x_batch, inducing_points, kernel_fn,
    inducing_mu, inducing_sigma_mat,
    batch_num, induce_num, class_num, kernel_scale
):
    batch_induced = concatenate([x_batch, inducing_points], axis=0)

    inducing_x = inducing_points
    inducing_y = zeros((induce_num, class_num))
    predict_fn = gradient_descent_mse_ensemble(kernel_fn, inducing_x, inducing_y)#, diag_reg=1e-4)
    _, B_B = predict_fn(x_test=x_batch, get="nngp", compute_cov=True)

    B_B *= kernel_scale

    kernel = kernel_fn(batch_induced, batch_induced, "nngp") * kernel_scale
    k_b_b, k_b_i, k_i_b, k_i_i = split_kernel(kernel, batch_num)
    k_i_i_inverse = inv(k_i_i + 1e-4 * eye(induce_num))

    A_B = matmul(k_b_i, k_i_i_inverse)
    # L = cholesky(k_i_i)
    # A_B_L = matmul(A_B, L)
    A_B_L = A_B
    A_B_L = kron_diag(A_B_L, n=class_num)
    B_B = kron_diag(B_B, n=class_num)

    mean = matmul(A_B_L, inducing_mu)
    covariance = matmul3(A_B_L, inducing_sigma_mat, A_B_L.T) + B_B
    covariance_L = cholesky(covariance)

    return mean, covariance_L
