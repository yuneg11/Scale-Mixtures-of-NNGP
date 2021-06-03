from jax import random
from jax import numpy as jnp
from jax.experimental import optimizers

from tqdm import tqdm

from . import data
from .ops import *
from .utils import *
from .classification_utils import *

from . import svgp
from . import svtp


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
    parser.add_argument("-trn",  "--train-num",       default=10000, type=int)
    parser.add_argument("-tsn",  "--test-num",        default=1000, type=int)
    parser.add_argument("-bs",   "--batch-size",      default=128, type=int)
    parser.add_argument("-in",   "--induce-num",      default=100, type=int)
    parser.add_argument("-sn",   "--sample-num",      default=100, type=int)
    parser.add_argument("-tssn", "--test-sample-num", default=10000, type=int)
    parser.add_argument("-a",    "--alpha",           default=4., type=float)
    parser.add_argument("-b",    "--beta",            default=4., type=float)
    parser.add_argument("-lr",   "--learning_rate",   default=0.001, type=float)
    parser.add_argument("-pi",   "--print-interval",  default=100, type=int)
    parser.add_argument("-ks",   "--kernel-scale",   default=1., type=float)
    parser.add_argument("-act",  "--activation",      default="relu", choices=["erf", "relu"])
    parser.add_argument("-wv",   "--w-variance",      default=1., type=float)
    parser.add_argument("-bv",   "--b-variance",      default=0., type=float)
    parser.add_argument("-lwv",  "--last-w-variance", default=1., type=float)
    parser.add_argument("-ti",   "--test-interval",   default=500, type=int)
    parser.add_argument("-dp",   "--depth",           default=4, type=int)
    parser.add_argument("-opt",  "--optimizer",       default="adam",  choices=["adam", "sgd"])
    parser.add_argument("-e",    "--steps",           default=20000, type=int)
    parser.add_argument("-s",    "--seed",            default=10, type=int)
    parser.add_argument("-nn",   "--no-normalize",    default=False, action="store_true")


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

    # Kernel
    if dataset in ["mnist", "cifar10", "cifar100"]:
        kernel_fn = get_cnn_kernel(depth, class_num, activation, sqrt(w_variance), sqrt(b_variance), sqrt(last_w_variance))
    elif dataset in ["iris"]:
        kernel_fn = get_mlp_kernel(depth, class_num, activation, sqrt(w_variance), sqrt(b_variance), sqrt(last_w_variance))
    else:
        raise ValueError("Unsupported dataset '{}'".format(dataset))

    # Init
    key = random.PRNGKey(10)

    inducing_points = x_train[:induce_num]
    inducing_mu = jnp.zeros(induce_num * class_num)
    inducing_sigma = jnp.ones(induce_num * class_num)

    if method == "svgp":
        train_vars = svgp.get_train_vars(inducing_mu, inducing_sigma, inducing_points)
        train_consts = (train_num, class_num, sample_num, induce_num, batch_size)
        test_consts = (test_num, class_num, test_sample_num, induce_num)
        test_nll_acc = svgp.test_nll_acc
    elif method == "svtp":
        invgamma_a = alpha
        invgamma_b = beta

        train_vars = svtp.get_train_vars(inducing_mu, inducing_sigma, inducing_points,
                                         invgamma_a, invgamma_b)
        train_consts = (alpha, beta, train_num, class_num, sample_num, induce_num, batch_size)
        test_consts = (test_num, class_num, test_sample_num, induce_num)
        test_nll_acc = svtp.test_nll_acc
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

    for i, (x_batch, y_batch) in tqdm(enumerate(train_batches), total=steps):
        key, split_key = random.split(key)
        train_params = get_params(opt_state)

        grads = grad_elbo_jit(x_batch, y_batch,
                                kernel_fn, kernel_scale,
                                *train_params,
                                *train_consts, split_key)

        opt_state = opt_update(i, grads, opt_state)

        if i % print_interval == 0:
            train_params = get_params(opt_state)
            n_elbo = negative_elbo_jit(x_batch, y_batch,
                                        kernel_fn, kernel_scale,
                                        *train_params,
                                        *train_consts, split_key)

            elbo_print = "{} / {}: nELBO = {:.6f}".format(i, len(train_batches), n_elbo)
            tqdm.write(elbo_print)

            if log_dir is not None:
                log_file.write(elbo_print + "\n")
                log_file.flush()

        if i % test_interval == 0:

            test_nll, test_acc = test_nll_acc(x_test, y_test, kernel_fn, kernel_scale,
                                              *train_params, *test_consts, key)

            nll_acc_print = "{} / {}: test_nll = {:.6f}, test_acc = {:.4f}".format(i, len(train_batches), test_nll, test_acc)
            tqdm.write(nll_acc_print)

            if log_dir is not None:
                log_file.write(nll_acc_print + "\n")
                log_file.flush()

    log_file.close()
