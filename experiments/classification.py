from collections import namedtuple

import numpy as np
import tensorflow_datasets as tfds

from jax import jit, grad, random
from jax import numpy as jnp
from jax.scipy.special import gammaln, digamma
from jax.experimental import optimizers

from neural_tangents import stax
from neural_tangents.predict import gradient_descent_mse_ensemble

from scipy import stats

from .utils import *


def add_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("classification",
                                   aliases=["cls"],
                                   help="classification",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=main)

    parser.add_argument("-d",  "--dataset", choices=["mnist", "iris", "test"])
    parser.add_argument("-log",  "--log-name",        required=False)
    parser.add_argument("-trn",  "--train-num",       default=10000, type=int)
    parser.add_argument("-tsn",  "--test-num",        default=1000, type=int)
    parser.add_argument("-bn",   "--batch-num",       default=32, type=int)
    parser.add_argument("-in",   "--induce-num",      default=100, type=int)
    parser.add_argument("-sn",   "--sample-num",      default=100, type=int)
    parser.add_argument("-tssn", "--test-sample-num", default=10000, type=int)
    parser.add_argument("-a",    "--alpha",           default=4., type=float)
    parser.add_argument("-b",    "--beta",            default=4., type=float)
    parser.add_argument("-lr",   "--learning_rate",   default=0.00001, type=float)
    parser.add_argument("-s",    "--epochs",          default=200000, type=int)
    parser.add_argument("-pi",   "--print_interval",  default=100, type=int)


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def get_dataset(name, train_num=None, test_num=None, seed=0):
    data_dir = "./data"
    if name != "test":
        ds_builder = tfds.builder(name)

    if name == "mnist":
        ds_train, ds_test = tfds.as_numpy(
            tfds.load(
                name,
                split=["train" + ("[:%d]" % train_num if train_num is not None else ""),
                    "test" + ("[:%d]" % test_num if test_num is not None else "")],
                batch_size=-1,
                as_dataset_kwargs={"shuffle_files": False},
                data_dir=data_dir,
            )
        )
        dataset = (ds_train["image"], ds_train["label"], ds_test["image"], ds_test["label"])
        x_train, y_train, x_test, y_test = dataset

        num_classes = ds_builder.info.features["label"].num_classes

    elif name == "iris":
        ds_train, = tfds.as_numpy(
            tfds.load(
                name,
                # split=["train" + ("[:%d]" % train_num if train_num is not None else "")],
                split=["train"],
                batch_size=-1,
                as_dataset_kwargs={"shuffle_files": False},
                data_dir=data_dir,
            )
        )
        x, y = ds_train["features"], ds_train["label"]
        x, y = permute_dataset(x, y, seed=109)
        x_train, y_train, x_test, y_test = x[:train_num], y[:train_num], x[test_num:], y[test_num:]

        num_classes = ds_builder.info.features["label"].num_classes

    elif name == "test":
        import gpflow

        num_classes = 3
        n = train_num + test_num

        lengthscales = 0.1
        jitter_eye = np.eye(n) * 1e-6
        x = np.random.rand(n, 1)
        kernel_se = gpflow.kernels.SquaredExponential(lengthscales=lengthscales)
        k = kernel_se(x) + jitter_eye
        f = np.random.multivariate_normal(mean=np.zeros(n), cov=k, size=(num_classes)).T
        y = np.argmax(f, 1).reshape(-1,).astype(int)

        x_train, y_train, x_test, y_test = x[:train_num], y[:train_num], x[test_num:], y[test_num:]

    else:
        raise KeyError("Unsupported dataset '{}'".format(name))

    y_train = _one_hot(y_train, num_classes)
    y_test = _one_hot(y_test, num_classes)

    x_train, y_train = permute_dataset(x_train, y_train, seed=seed)

    x_train = (x_train - jnp.mean(x_train)) / jnp.std(x_train)
    x_test = (x_test - jnp.mean(x_train)) / jnp.std(x_train)

    # TODO: DEBUG START
    # y_train_0 = jnp.sum(y_train[:, [0, 2, 4, 6, 8]], axis=1, keepdims=True)
    # y_train_1 = jnp.sum(y_train[:, [1, 3, 5, 7, 9]], axis=1, keepdims=True)
    # y_train = jnp.concatenate([y_train_0, y_train_1], axis=1)
    # DEBUG END

    return x_train, y_train, x_test, y_test


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


def get_mlp_kernel(depth, class_num):
    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(512, W_std = 4., b_std = 1.))
        layers.append(stax.Relu())
    layers.append(stax.Dense(class_num, W_std=1))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))

    return  kernel_fn


def mean_covariance(
    x_batch, inducing_points, kernel_fn,
    inducing_mu, inducing_sigma_mat,
    batch_num, induce_num, class_num,
):
    batch_induced = jnp.concatenate([x_batch, inducing_points], axis=0)

    inducing_x = inducing_points
    inducing_y = jnp.zeros((induce_num, class_num))
    predict_fn = gradient_descent_mse_ensemble(kernel_fn, inducing_x, inducing_y)#, diag_reg=1e-4)
    _, B_B = predict_fn(x_test=x_batch, get="nngp", compute_cov=True)

    kernel = kernel_fn(batch_induced, batch_induced, "nngp")
    k_b_b, k_b_i, k_i_b, k_i_i = split_kernel(kernel, batch_num)
    k_i_i_inverse = jnp.linalg.inv(k_i_i + 1e-4 * jnp.eye(induce_num))

    A_B = matmul2(k_b_i, k_i_i_inverse)
    L = jnp.linalg.cholesky(k_i_i)
    A_B_L = matmul2(A_B, L)
    A_B_L = kron_diag(A_B_L, n=class_num)
    B_B = kron_diag(B_B, n=class_num)

    mean = matmul2(A_B_L, inducing_mu)
    covariance = matmul3(A_B_L, inducing_sigma_mat, A_B_L.T) + B_B
    covariance_L = jnp.linalg.cholesky(covariance)

    return mean, covariance_L


def g_sample_f_b(sample_num, batch_num, class_num, key):
    _, key_normal = random.split(key)

    mean = jnp.zeros(batch_num * class_num)
    covariance = jnp.eye(batch_num * class_num)
    sampled_f = random.multivariate_normal(key_normal, mean, covariance, shape=(sample_num,))

    return sampled_f


def t_sample_f_b(invgamma_a, invgamma_b, sample_num, batch_num, class_num, key):
    _, key_gamma, key_normal = random.split(key, 3)

    sampled_f = g_sample_f_b(sample_num, batch_num, class_num, key_normal)

    gamma_pure = random.gamma(key_gamma, a=invgamma_a)
    gamma_rho = gamma_pure / invgamma_b
    invgamma = 1 / gamma_rho  # invgamma ~ invgamma(a = nu_q/2, scale = rho_q/2)
    sigma = jnp.sqrt(invgamma)
    sampled_f = sampled_f * sigma

    return sampled_f


def g_kl_divergence(inducing_mu, inducing_sigma_mat, class_num, induce_num):
    kl = 1 / 2 * (-jnp.log(jnp.linalg.det(inducing_sigma_mat)) - induce_num * class_num \
       + jnp.trace(inducing_sigma_mat) + jnp.matmul(inducing_mu, inducing_mu))
    return kl


def t_kl_divergence(inducing_mu, inducing_sigma_mat, invgamma_a, invgamma_b, alpha, beta, class_num, induce_num):
    kl = 1 / 2 * (-jnp.log(jnp.linalg.det(inducing_sigma_mat)) - induce_num * class_num \
       + jnp.trace(inducing_sigma_mat) + invgamma_a / invgamma_b * jnp.matmul(inducing_mu, inducing_mu)) \
       + alpha * jnp.log(invgamma_b / beta) - gammaln(invgamma_a) + gammaln(alpha) \
       + (invgamma_a - alpha) * digamma(invgamma_a) + invgamma_a / invgamma_b * (beta - invgamma_b)
    return kl


def log_likelihood(label, sampled_f, batch_num, class_num, sample_num, train_num):
    sampled_f = jnp.transpose(sampled_f.reshape(sample_num, class_num, batch_num), axes=(0, 2, 1))

    # true_label_softmax = robust_max_values(sampled_f, label, class_num)
    # ll = true_label_softmax * train_num / batch_num / sample_num

    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_values(sampled_f_softmax, label)
    ll = train_num * jnp.mean(jnp.mean(true_label_softmax, axis=1))

    return ll


def log_likelihood2(label, sampled_f, class_num, sample_num, test_num):
    sampled_f = jnp.transpose(sampled_f.reshape(sample_num, class_num, test_num), axes=(0, 2, 1))

    # TODO
    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_values(sampled_f_softmax, label)
    ll = jnp.sum(logsumexp(true_label_softmax.T) - jnp.log(sample_num)) / test_num

    return ll


def g_ELBO(
    x_batch, y_batch, kernel_fn,
    inducing_points, inducing_mu, inducing_sigma,
    nums, seed=0,
):
    sample_num = nums.sample_num
    batch_num = nums.batch_num
    induce_num = nums.induce_num
    train_num = nums.train_num
    class_num = nums.class_num

    inducing_sigma_mat = jnp.diag(inducing_sigma)

    mean, covariance_L = mean_covariance(x_batch, inducing_points, kernel_fn,
                                         inducing_mu, inducing_sigma_mat,
                                         batch_num, induce_num, class_num)

    key = random.PRNGKey(seed)
    sampled_f = g_sample_f_b(sample_num, batch_num, class_num, key)
    sampled_f = (mean.reshape(-1, 1) + matmul2(covariance_L, sampled_f.T)).T

    ll = log_likelihood(y_batch, sampled_f, batch_num, class_num, sample_num, train_num)
    kl = g_kl_divergence(inducing_mu, inducing_sigma_mat, class_num, induce_num)

    elbo = (ll - kl) / train_num
    return -elbo


def t_ELBO(
    x_batch, y_batch, kernel_fn,
    inducing_points, inducing_mu, inducing_sigma,
    invgamma_a, invgamma_b, alpha, beta,
    nums, seed=0,
):
    sample_num = nums.sample_num
    batch_num = nums.batch_num
    induce_num = nums.induce_num
    train_num = nums.train_num
    class_num = nums.class_num

    inducing_sigma_mat = jnp.diag(inducing_sigma)

    mean, covariance_L = mean_covariance(x_batch, inducing_points, kernel_fn,
                                         inducing_mu, inducing_sigma_mat,
                                         batch_num, induce_num, class_num)

    key = random.PRNGKey(seed)
    sampled_f = t_sample_f_b(invgamma_a, invgamma_b, sample_num, batch_num, class_num, key)
    sampled_f = (mean.reshape(-1, 1) + matmul2(covariance_L, sampled_f.T)).T

    ll = log_likelihood(y_batch, sampled_f, batch_num, class_num, sample_num, train_num)
    kl = t_kl_divergence(inducing_mu, inducing_sigma_mat, invgamma_a, invgamma_b, alpha, beta, class_num, induce_num)

    elbo = (ll - kl) / train_num
    return -elbo


def main(dataset, log_name, train_num, test_num, batch_num, induce_num, sample_num,
         test_sample_num, alpha, beta, learning_rate, epochs, print_interval, **kwargs):

    alpha = None
    beta = None

    seed = 10

    if log_name is not None:
        log_file = open(log_name, "w")

    depth = 3

    # Dataset
    if dataset == "iris":
        train_num   = 120
        test_num    = 30
    elif dataset == "test":
        train_num   = 850
        test_num    = 150

    batch_num       = 32
    induce_num      = 100
    sample_num      = 10
    test_sample_num = 10000
    learning_rate   = 0.000001
    epochs          = 200000
    print_interval  = 100
    depth           = 4

    x_train, y_train, x_test, y_test = get_dataset(dataset, train_num, test_num, seed=seed)
    class_num = y_train.shape[1]

    # Kernel
    if dataset == "mnist":
        kernel_fn = get_cnn_kernel(depth=depth, class_num=class_num)
    elif dataset == "iris" or dataset == "test":
        kernel_fn = get_mlp_kernel(depth=depth, class_num=class_num)

    # Init
    key = random.PRNGKey(10)
    inducing_points = random.normal(key, shape=(induce_num, *x_train.shape[1:]))

    # inducing_points = x_train[:induce_num]
    inducing_mu = jnp.zeros(induce_num * class_num)
    inducing_sigma = jnp.ones(induce_num * class_num)

    train_params = (inducing_points, inducing_mu, inducing_sigma)

    # TODO: Refactoring
    nums = namedtuple("nums", [
        "train_num", "test_num", "batch_num", "induce_num",
        "sample_num", "test_sample_num", "class_num",
    ])(train_num, test_num, batch_num, induce_num, sample_num, test_sample_num, class_num)

    # Grads

    ELBO_jit = g_ELBO
    # ELBO_jit = jit(g_ELBO, static_argnums=(2, 6, 7))
    grad_elbo = grad(ELBO_jit, argnums=(3, 4, 5))
    # grad_elbo = jit(grad_elbo, static_argnums=(2, 6, 7))

    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(train_params)

    # Training
    train_batches = TrainBatch(x_train, y_train, batch_size=batch_num, epochs=epochs+1, seed=seed)

    for i, (x_batch, y_batch) in enumerate(train_batches):
        train_params = get_params(opt_state)

        if i % print_interval == 0:
            elbo = ELBO_jit(x_batch, y_batch, kernel_fn, *train_params, nums)
            print("{} / {}: ELBO = {:.7f}".format(i, len(train_batches) - 1, elbo))
            if log_name is not None:
                log_file.write("{} / {}: ELBO = {:.7f}\n".format(i, len(train_batches) - 1, elbo))
                log_file.flush()

        opt_state = opt_update(i, grad_elbo(x_batch, y_batch, kernel_fn, *train_params, nums), opt_state)

    # Test
    inducing_points, inducing_mu, inducing_sigma = get_params(opt_state)

    induced_test = jnp.concatenate([inducing_points, x_test], axis=0)
    kernel = kernel_fn(induced_test, induced_test, "nngp")

    kernel_i_i, kernel_i_t, kernel_t_i, kernel_t_t = split_kernel(kernel, induce_num)
    kernel_i_i_inverse = jnp.linalg.inv(kernel_i_i + 1e-4 * jnp.eye(induce_num))

    L_induced = jnp.linalg.cholesky(kernel_i_i)

    A = matmul3(kernel_t_i, kernel_i_i_inverse, L_induced)
    A = kron_diag(A, n=class_num)

    L_induced = kron_diag(L_induced, n=class_num)
    L_mu = matmul2(L_induced, inducing_mu)

    inducing_x = inducing_points
    inducing_y = jnp.transpose(L_mu.reshape(-1, induce_num))
    predict_fn = gradient_descent_mse_ensemble(kernel_fn, inducing_x, inducing_y, diag_reg=1e-4)

    test_mean, test_covariance = predict_fn(x_test=x_test, get="nngp", compute_cov=True)
    test_mean = test_mean.T.flatten()
    test_covariance = kron_diag(test_covariance, n=class_num)

    inducing_sigma = jnp.diag(inducing_sigma)
    A_sigma = matmul3(A, inducing_sigma, A.T)
    test_sigma = A_sigma + test_covariance

    _, key_test = random.split(key)

    test_f_sample = random.multivariate_normal(
        key=key_test,
        mean=test_mean,
        conv=test_sigma,
        shape=(test_sample_num,),
    )

    test_nll = -log_likelihood2(y_test, test_f_sample, class_num, sample_num, test_num)

    print(test_nll)

    if log_name is not None:
        log_file.write(test_nll)
        log_file.close()


def t_main(dataset, log_name, train_num, test_num, batch_num, induce_num, sample_num,
         test_sample_num, alpha, beta, learning_rate, epochs, print_interval, **kwargs):

    seed = 10

    if log_name is not None:
        log_file = open(log_name, "w")

    depth = 3

    # Dataset
    if dataset == "iris":
        train_num       = 120
        test_num        = 30
        batch_num       = 32
        induce_num      = 100
        sample_num      = 100
        test_sample_num = 10000
        alpha           = 2.
        beta            = 2.
        learning_rate   = 0.000001
        epochs          = 200000
        print_interval  = 1
        depth           = 1
    elif dataset == "test":
        train_num       = 120
        test_num        = 30
        batch_num       = 32
        induce_num      = 100
        sample_num      = 100
        test_sample_num = 10000
        alpha           = 2.
        beta            = 2.
        learning_rate   = 0.000001
        epochs          = 200000
        print_interval  = 1
        depth           = 1

    x_train, y_train, x_test, y_test = get_dataset(dataset, train_num, test_num, seed=seed)
    class_num = y_train.shape[1]

    # Kernel
    if dataset == "mnist":
        kernel_fn = get_cnn_kernel(depth=depth, class_num=class_num)
    elif dataset == "iris" or dataset == "test":
        kernel_fn = get_mlp_kernel(depth=depth, class_num=class_num)

    # Init
    key = random.PRNGKey(10)
    inducing_points = random.normal(key, shape=(induce_num, *x_train.shape[1:]))

    # inducing_points = x_train[:induce_num]
    inducing_mu = jnp.zeros(induce_num * class_num)
    inducing_sigma = jnp.ones(induce_num * class_num)
    invgamma_a = alpha
    invgamma_b = beta

    train_params = (inducing_points, inducing_mu, inducing_sigma, invgamma_a, invgamma_b)

    # TODO: Refactoring
    nums = namedtuple("nums", [
        "train_num", "test_num", "batch_num", "induce_num",
        "sample_num", "test_sample_num", "class_num",
    ])(train_num, test_num, batch_num, induce_num, sample_num, test_sample_num, class_num)

    # Grads
    ELBO_jit = jit(t_ELBO, static_argnums=(2, 8, 9, 10, 11))
    # ELBO_jit = ELBO
    grad_elbo = grad(ELBO_jit, argnums=(3, 4, 5, 6, 7))
    grad_elbo = jit(grad_elbo, static_argnums=(2, 8, 9, 10, 11))

    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(train_params)

    # Training
    train_batches = TrainBatch(x_train, y_train, batch_size=batch_num, epochs=epochs+1, seed=seed)

    for i, (x_batch, y_batch) in enumerate(train_batches):
        train_params = get_params(opt_state)

        if i % print_interval == 0:
            elbo = ELBO_jit(x_batch, y_batch, kernel_fn, *train_params, alpha, beta, nums)
            print("{} / {}: ELBO = {:.7f}".format(i, len(train_batches) - 1, elbo))
            if log_name is not None:
                log_file.write("{} / {}: ELBO = {:.7f}\n".format(i, len(train_batches) - 1, elbo))
                log_file.flush()

        opt_state = opt_update(i, grad_elbo(x_batch, y_batch, kernel_fn, *train_params, alpha, beta, nums), opt_state)

    # Test
    inducing_points, inducing_mu, inducing_sigma, invgamma_a, invgamma_b = get_params(opt_state)

    induced_test = jnp.concatenate([inducing_points, x_test], axis=0)
    kernel = kernel_fn(induced_test, induced_test, "nngp")

    kernel_i_i, kernel_i_t, kernel_t_i, kernel_t_t = split_kernel(kernel, induce_num)
    kernel_i_i_inverse = jnp.linalg.inv(kernel_i_i + 1e-4 * jnp.eye(induce_num))

    L_induced = jnp.linalg.cholesky(kernel_i_i)

    A = matmul3(kernel_t_i, kernel_i_i_inverse, L_induced)
    A = kron_diag(A, n=class_num)

    L_induced = kron_diag(L_induced, n=class_num)
    L_mu = matmul2(L_induced, inducing_mu)

    inducing_x = inducing_points
    inducing_y = jnp.transpose(L_mu.reshape(-1, induce_num))
    predict_fn = gradient_descent_mse_ensemble(kernel_fn, inducing_x, inducing_y, diag_reg=1e-4)

    test_mean, test_covariance = predict_fn(x_test=x_test, get="nngp", compute_cov=True)
    test_mean = test_mean.T.flatten()
    test_covariance = kron_diag(test_covariance, n=class_num)

    inducing_sigma = jnp.diag(inducing_sigma)
    A_sigma = matmul3(A, inducing_sigma, A.T)
    test_sigma = invgamma_b / invgamma_a * (A_sigma + test_covariance)

    test_f_sample = stats.multivariate_t.rvs(
        loc=test_mean,
        shape=test_sigma,
        df=2 * invgamma_a,
        size=(test_sample_num,),
    )

    test_nll = -log_likelihood2(y_test, test_f_sample, class_num, sample_num, test_num)

    print(test_nll)

    if log_name is not None:
        log_file.write(test_nll)
        log_file.close()
