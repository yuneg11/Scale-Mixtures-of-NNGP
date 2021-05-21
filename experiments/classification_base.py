# from collections import namedtuple
from functools import partial

from jax._src.numpy.lax_numpy import var
import numpy as np
import tensorflow_datasets as tfds

import jax
from jax import jit, grad, random
from jax import numpy as jnp
from jax.scipy.special import gammaln, digamma
from jax.experimental import optimizers

from neural_tangents import stax
from neural_tangents.predict import gradient_descent_mse_ensemble

from scipy import stats

from tqdm import tqdm

from . import data
from .utils import *


def add_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("classification_test",
                                   aliases=["ct"],
                                   help="cls_test",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=main)

    # parser.add_argument("-d",  "--dataset", choices=["mnist", "iris", "test"])
    # parser.add_argument("-log",  "--log-name",        required=False)
    # parser.add_argument("-trn",  "--train-num",       default=10000, type=int)
    # parser.add_argument("-tsn",  "--test-num",        default=1000, type=int)
    # parser.add_argument("-bn",   "--batch-num",       default=32, type=int)
    # parser.add_argument("-in",   "--induce-num",      default=100, type=int)
    # parser.add_argument("-sn",   "--sample-num",      default=100, type=int)
    # parser.add_argument("-tssn", "--test-sample-num", default=10000, type=int)
    # parser.add_argument("-a",    "--alpha",           default=4., type=float)
    # parser.add_argument("-b",    "--beta",            default=4., type=float)
    # parser.add_argument("-lr",   "--learning_rate",   default=0.00001, type=float)
    # parser.add_argument("-s",    "--epochs",          default=200000, type=int)
    # parser.add_argument("-pi",   "--print_interval",  default=100, type=int)


def get_cnn_kernel(depth, class_num):
    layers = []
    for _ in range(depth):
        layers.append(stax.Conv(1, (3, 3), (1, 1), 'SAME'))
        layers.append(stax.Relu())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(class_num, W_std=1))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return  kernel_fn


def get_mlp_kernel(depth, class_num, W_std=1., b_std=1.):
    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(512, W_std=W_std, b_std=b_std))
        layers.append(stax.Relu())
    layers.append(stax.Dense(class_num, W_std=1))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return  kernel_fn


def get_rbf_kernel_fn():
    rbf = lambda x, v, l: v * jnp.exp(-0.5 * ((x - x.T) ** 2) / l)
    return rbf


def get_kernel_fn(name, *args, **kwargs):
    if name == "cnn":
        return get_cnn_kernel(*args, **kwargs)
    elif name == "mlp":
        return get_mlp_kernel(*args, **kwargs)
    elif name == "rbf":
        return get_rbf_kernel_fn()
    else:
        raise KeyError("Unsupported kernel '{}'".format(name))
    
# @partial(jit, static_argnums=(2,3))
def gaussian_kl_divergence(inducing_mu, inducing_sigma, inducing_num, class_num):
    kl = (log(det(inducing_sigma)) - (inducing_num * class_num)
            + trace(inducing_sigma) + matmul(inducing_mu, inducing_mu)) / 2
    return kl

# @partial(jit, static_argnums=(4, 5))
def calculate_mean_variance(batch_x, inducing_points, inducing_mu, inducing_sigma, inducing_num, kernel_fn, kernel_v, kernel_l):
    induced_batch = jnp.concatenate([inducing_points, batch_x], axis = 0)
    kernel = kernel_fn(induced_batch, kernel_v, kernel_l)
    k_i_i, k_i_b, k_b_i, k_b_b = split_kernel(kernel, inducing_num)
    k_i_i_inverse = jnp.linalg.inv(k_i_i)
    A = jnp.matmul(k_b_i,k_i_i_inverse)
    D = k_b_b - matmul3(k_b_i,k_i_i_inverse,k_i_b)
    mean = jnp.matmul(A, inducing_mu)
    cov = matmul3(A, inducing_sigma, A.T) + D
    return mean, cov


# @partial(jit, static_argnums=(4))
def loglikelihood(batch_y, mean, variance, gaussian_v, train_num):
    # logpdf = jax.scipy.stats.multivariate_normal.logpdf(batch_y, mean, variance)
    # logpdf = jax.scipy.stats.norm.logpdf(batch_y, mean, jnp.sqrt(jnp.diag(variance)))
    
    return train_num * jnp.mean(-0.5 * (np.log(2 * np.pi) + jnp.log(gaussian_v) + ((batch_y - mean) ** 2 + jnp.diag(variance)) / gaussian_v))
    
    
    avg = train_num * np.mean(logpdf)
    return avg
    
def ELBO(inducing_points, inducing_mu, inducing_sigma, gaussian_v, kernel_v, kernel_l, batch_x, batch_y, batch_num, inducing_num, train_num, kernel_fn):
    inducing_sigma = jnp.diag(inducing_sigma)
    mean, variance = calculate_mean_variance(batch_x, inducing_points, inducing_mu, inducing_sigma, inducing_num, kernel_fn, kernel_v, kernel_l)
    log_likelihood = loglikelihood(batch_y, mean, variance, gaussian_v, train_num)
    kl = gaussian_kl_divergence(inducing_mu, inducing_sigma, inducing_num, 1)
    elbo = log_likelihood - kl
    return -elbo


def predict(test_x, kernel_fn, gaussian_v, kernel_v, kernel_l, inducing_points, inducing_mu, inducing_sigma, test_num, inducing_num):
    inducing_sigma = jnp.diag(inducing_sigma)
    mu, covariance = calculate_mean_variance(test_x, inducing_points, inducing_mu, inducing_sigma, inducing_num, kernel_fn, kernel_v, kernel_l)
    covariance += gaussian_v*np.eye(test_num)
    std = np.sqrt(np.diag(covariance))
    return mu, std




def plot(filename, train_x, train_y, kernel_fn, gaussian_v, kernel_v, kernel_l, inducing_points, inducing_mu, inducing_sigma, test_num, inducing_num):
    test_num = 100
    test_x = np.linspace(-1, +1, test_num)[:, None]
    predict_mu, predict_sigma = predict(test_x, kernel_fn, gaussian_v, kernel_v, kernel_l, inducing_points, inducing_mu, inducing_sigma, test_num, inducing_num)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.title(f"Prediction {filename}", fontsize=11)
    plt.plot(train_x, train_y, "x", label="Training points", alpha=0.2)
    (line,) = plt.plot(test_x, predict_mu, lw=2.5, label="Mean of predictive posterior")
    col = line.get_color()
    plt.fill_between(
        test_x[:, 0],
        (predict_mu - 2 * predict_sigma ** 0.5),
        (predict_mu + 2 * predict_sigma ** 0.5),
        color=col,
        alpha=0.2,
        lw=1.5,
    )
    plt.legend(loc="lower right", fontsize=11)
    plt.savefig(filename)
    plt.close()



def main(**kwargs):
    dataset = "test_cls"
    train_num = 10000
    test_num = 0
    batch_num = 100
    inducing_num = 15
    learning_rate = 0.00001
    gaussian_v = 1.
    kernel_v = 1.
    kernel_l = 1.
    
    steps          = 30000
    print_interval = 100
    plot_interval = 100
    
    
    train_test_data = data.get_dataset(dataset, train_num=train_num, test_num=test_num)
    train_x, train_y, test_x, test_y = train_test_data
    
    kernel_fn = get_kernel_fn('rbf')
    
    
    
    idx = [int(i) for i in list(np.linspace(0, train_num, inducing_num, endpoint=False))]
    inducing_points = train_x[idx, ...].copy()

    # inducing_points = train_x[:inducing_num]
    inducing_mu = np.zeros(inducing_num)
    inducing_sigma = np.ones(inducing_num)
    
    
    # train_batches = iter(TrainBatch(train_x, train_y, batch_size=batch_num, epochs=steps+1, seed=109))
    
    # batch_x, batch_y = next(train_batches)
    
    # elbo = ELBO(inducing_points, inducing_mu, inducing_sigma, kernel_parameter, batch_x, batch_y, batch_num, inducing_num, train_num, kernel_fn)
    # print(elbo)

    elbo_jit = jit(ELBO, static_argnums = (8,9,10, 11))
    grad_elbo = grad(ELBO, argnums = (0,1,2,3,4, 5))
    grad_elbo = jit(grad_elbo, static_argnums = (8,9,10, 11))

    train_params = (inducing_points, inducing_mu, inducing_sigma, gaussian_v, kernel_v, kernel_l)

        
    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(train_params)

    # Training
    train_batches = TrainBatch(train_x, train_y, batch_size=batch_num, epochs=steps+1, seed=109)

    for i, (batch_x, batch_y) in tqdm(enumerate(train_batches), total=len(train_batches)-1):
        train_params = get_params(opt_state)

        if i % print_interval == 0:
            elbo = elbo_jit(*train_params, batch_x, batch_y, batch_num, inducing_num, train_num, kernel_fn)
            # print("{} / {}: ELBO = {:.7f}".format(i, len(train_batches) - 1, elbo))
            tqdm.write("{} / {}: ELBO = {:.7f}".format(i, len(train_batches) - 1, elbo))
            # if log_name is not None:
            #     log_file.write("{} / {}: ELBO = {:.7f}\n".format(i, len(train_batches) - 1, elbo))
            #     log_file.flush()

        opt_state = opt_update(i, grad_elbo(*train_params, batch_x, batch_y, batch_num, inducing_num, train_num, kernel_fn), opt_state)
        
        if i % plot_interval == 0:
            inducing_points, inducing_mu, inducing_sigma, gaussian_v, kernel_v, kernel_l = get_params(opt_state)
            plot(f"testlrd{i}.png", train_x, train_y, kernel_fn, gaussian_v, kernel_v, kernel_l, inducing_points, inducing_mu, inducing_sigma, test_num, inducing_num)
