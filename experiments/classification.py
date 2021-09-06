from jax import random
from jax import numpy as jnp
from jax.experimental import optimizers
import math

from tqdm import tqdm

from . import data
from .ops import *
from .utils import *
from .classification_utils import *

from . import svgp
from . import svtp

import pdb


def softplus(x):
    return log(1 + exp(x))


def softplus_inv(x):
    return log(exp(x) - 1.)


def add_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("classification",
                                   aliases=["cls"],
                                   help="classification",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=main)

    parser.add_argument("method",                     choices=["svgp", "svtp"])
    parser.add_argument("-d",    "--dataset",         choices=data.classification_datasets)
    parser.add_argument("-log",  "--log-dir",         required=False)
    parser.add_argument("-trn",  "--train-num",       default=50000, type=int)
    parser.add_argument("-tsn",  "--test-num",        default=10000, type=int)
    parser.add_argument("-bs",   "--batch-size",      default=128, type=int)
    parser.add_argument("-in",   "--induce-num",      default=100, type=int)
    parser.add_argument("-sn",   "--sample-num",      default=100, type=int)
    parser.add_argument("-tssn", "--test-sample-num", default=10000, type=int)
    parser.add_argument("-a",    "--alpha",           default=2., type=float)
    parser.add_argument("-b",    "--beta",            default=2., type=float)
    parser.add_argument("-lr",   "--learning_rate",   default=0.001, type=float)
    parser.add_argument("-pi",   "--print-interval",  default=100, type=int)
    parser.add_argument("-ks",   "--kernel-scale",    default=1., type=float)
    parser.add_argument("-act",  "--activation",      default="relu", choices=["erf", "relu"])
    parser.add_argument("-wv",   "--w-variance",      default=1., type=float)
    parser.add_argument("-bv",   "--b-variance",      default=1e-8, type=float)
    parser.add_argument("-lv",   "--last-w-variance", default=1., type=float)
    parser.add_argument("-ti",   "--test-interval",   default=500, type=int)
    parser.add_argument("-dp",   "--depth",           default=4, type=int)
    parser.add_argument("-opt",  "--optimizer",       default="adam",  choices=["adam", "sgd"])
    parser.add_argument("-e",    "--steps",           default=30000, type=int)
    parser.add_argument("-s",    "--seed",            default=10, type=int)
    parser.add_argument("-nn",   "--no-normalize",    default=False, action="store_true")
    parser.add_argument("-km",   "--kmeans",        default=False, action="store_true")


from sklearn.cluster import KMeans

def get_inducing(X, Y, inducing_num, class_num):
    inducings = []
    inducing_per_class = inducing_num // class_num
    for class_idx in range(class_num):
        xc = X[jnp.argmax(Y, axis=1) == class_idx]
        xc = xc.reshape(xc.shape[0], -1)
        kmeans = KMeans(n_clusters=inducing_per_class).fit(xc)
        inducings.append(kmeans.cluster_centers_.reshape(kmeans.cluster_centers_.shape[0], *X.shape[1:]))
    return jnp.vstack(inducings)


def main(method, dataset, train_num, test_num, induce_num, no_normalize, seed, depth, log_dir,
         alpha, beta, optimizer, learning_rate, steps, batch_size,
         print_interval, sample_num, kernel_scale, test_sample_num, test_interval,
         w_variance, b_variance, last_w_variance, activation, **kwargs):

    if log_dir is not None:
        if method == "svgp":
            log_name =  "/{}-{}-{}-{}.txt".format(depth, w_variance, b_variance, last_w_variance)
        elif method == "svtp":
            log_name = "/{}-{}-{}-{}-{}-{}.txt".format(depth, w_variance, b_variance, last_w_variance, alpha, beta)
        log_file = open(log_dir + log_name, "w")

    # Dataset
    x_train, y_train, x_test, y_test = data.get_dataset(
        dataset,
        train_num=train_num,
        test_num=test_num,
        normalize=(not no_normalize),
        seed=seed
    )
    class_num = y_train.shape[1]

    w_sigma = softplus_inv(math.sqrt(w_variance) + 1e-8)
    b_sigma = softplus_inv(math.sqrt(b_variance) + 1e-8)
    last_w_sigma = softplus_inv(math.sqrt(last_w_variance) + 1e-8)

    # Kernel
    if "mnist" in dataset or "cifar" in dataset or "svhn" in dataset:
        def get_kernel_fn(w_sigma, b_sigma, last_w_sigma):
            ws = softplus(w_sigma)
            bs = softplus(b_sigma)
            ls = softplus(last_w_sigma)
            # return get_cnn_kernel(depth, class_num, activation, ws, bs, ls)
            return get_cnn_kernel(depth, class_num, activation, ws, bs, ls)
    elif dataset in ["iris"]:
        def get_kernel_fn(w_sigma, b_sigma, last_w_sigma):
            ws = softplus(w_sigma)
            bs = softplus(b_sigma)
            ls = softplus(last_w_sigma)
            return get_mlp_kernel(depth, class_num, activation, ws, bs, ls)
    else:
        raise ValueError("Unsupported dataset '{}'".format(dataset))

    # Init
    key = random.PRNGKey(10)

    if kwargs["kmeans"]:
        inducing_points = get_inducing(x_train, y_train, induce_num, class_num)
    else:
        inducing_points = x_train[:induce_num]

    inducing_mu = jnp.zeros(induce_num * class_num)
    inducing_sigma = jnp.full(induce_num * class_num, softplus_inv(1e-6))

    if method == "svgp":
        train_vars = svgp.get_train_vars(inducing_mu, inducing_sigma, inducing_points, w_sigma, b_sigma, last_w_sigma)
        train_consts = (train_num, class_num, sample_num, induce_num, batch_size)
        test_consts = (test_num, class_num, test_sample_num, induce_num)
        test_nll_acc = svgp.test_nll_acc
        svgp.get_kernel_fn = get_kernel_fn
    elif method == "svtp":
        invgamma_a = softplus_inv(alpha + 1e-6)
        invgamma_b = softplus_inv(beta + 1e-6)

        train_vars = svtp.get_train_vars(inducing_mu, inducing_sigma, inducing_points,
                                         invgamma_a, invgamma_b, w_sigma, b_sigma, last_w_sigma)
        train_consts = (alpha, beta, train_num, class_num, sample_num, induce_num, batch_size)
        test_consts = (test_num, class_num, test_sample_num, induce_num)
        test_nll_acc = svtp.test_nll_acc
        svtp.get_kernel_fn = get_kernel_fn
    else:
        raise ValueError("Unsupported method '{}'".format(method))

    if optimizer == "adam":
        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    elif optimizer == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    else:
        raise ValueError("Unsupported optimizer '{}'".format(optimizer))

    train_params, negative_elbo_jit, grad_elbo_jit = train_vars
    opt_state = opt_init(train_params)

    # Training
    train_batches = TrainBatch(x_train, y_train, batch_size, steps, seed)

    for i, (x_batch, y_batch) in tqdm(enumerate(train_batches), total=steps, ncols=0):
        key, split_key = random.split(key)
        train_params = get_params(opt_state)

        grads = grad_elbo_jit(x_batch, y_batch, get_kernel_fn, kernel_scale, *train_params, *train_consts, split_key)

        if any([jnp.isnan(t).any() for i, t in enumerate(grads) if i != len(grads) - 2]):
            nelbo = negative_elbo_jit(x_batch, y_batch, get_kernel_fn, kernel_scale, *train_params, *train_consts, split_key)
            print(f"\nNaN occurred in gradients, nelbo: {nelbo}")
            print([jnp.isnan(t).any().item() for t in grads])
            pdb.set_trace()

        opt_state = opt_update(i, grads, opt_state)

        if i % print_interval == 0:
            train_params = get_params(opt_state)
            n_elbo = negative_elbo_jit(x_batch, y_batch, get_kernel_fn, kernel_scale, *train_params, *train_consts, split_key)

            ws = softplus(train_params[-3])
            bs = softplus(train_params[-2])
            ls = softplus(train_params[-1])

            elbo_print = "{} / {}: nELBO = {:.6f}".format(i, len(train_batches), n_elbo)
            # tqdm.write(elbo_print)
            if method == "svtp":
                ia = softplus(train_params[3])
                ib = softplus(train_params[4])
                tqdm.write(elbo_print + f"  ws: {ws:.4f}, bs: {bs:.4E}, ls: {ls:.4f}, a: {ia:.4f}, b: {ib:.4f}")
            else:
                tqdm.write(elbo_print + f"  ws: {ws:.4f}, bs: {bs:.4E}, ls: {ls:.4f}")

            if log_dir is not None:
                log_file.write(elbo_print + "\n")
                log_file.flush()

        if i % test_interval == 0 or i == steps - 1:
            test_nll, test_acc = test_nll_acc(x_test, y_test, get_kernel_fn, kernel_scale, *train_params, *test_consts, key)

            nll_acc_print = "{} / {}: test_nll = {:.6f}, test_acc = {:.4f}".format(i, len(train_batches), test_nll, test_acc)
            tqdm.write(nll_acc_print)

            if log_dir is not None:
                log_file.write(nll_acc_print + "\n")
                log_file.flush()

    if log_dir is not None:
        log_file.close()
