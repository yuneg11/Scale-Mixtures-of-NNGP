from collections import namedtuple

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

from .utils import *


def add_g_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("classification-g",
                                   aliases=["clsg"],
                                   help="classification-g",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=g_main)

    parser.add_argument("-d",     "--dataset",        choices=["mnist", "iris", "test"])
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
    parser.add_argument("-i",    "--inducing",        default=False, action="store_true")
    parser.add_argument("-k",    "--kernel-scale",    default=1., type=float)


def add_t_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("classification-t",
                                   aliases=["clst"],
                                   help="classification-t",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=t_main)

    parser.add_argument("-d",     "--dataset",        choices=["mnist", "iris", "test"])
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
    parser.add_argument("-i",    "--inducing",        default=False, action="store_true")
    parser.add_argument("-k",    "--kernel-scale",    default=1., type=float)


def get_cnn_kernel(depth, class_num):
    layers = []
    for _ in range(depth):
        layers.append(stax.Conv(1, (3, 3), (1, 1), 'SAME'))
        layers.append(stax.Relu())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(class_num, W_std=1))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2, ))

    return  kernel_fn


def get_mlp_kernel(depth, class_num, W_std=1., b_std=1.):
    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(512, W_std=W_std, b_std=b_std))
        layers.append(stax.Relu())
    layers.append(stax.Dense(class_num, W_std=1))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2, ))

    return  kernel_fn


# def mean_covariance(
#     x_batch, inducing_points, kernel_fn,
#     inducing_mu, inducing_sigma_mat,
#     batch_num, induce_num, class_num,
# ):
#     batch_induced = jnp.concatenate([x_batch, inducing_points], axis=0)

#     inducing_x = inducing_points
#     inducing_y = jnp.zeros((induce_num, class_num))
#     predict_fn = gradient_descent_mse_ensemble(kernel_fn, inducing_x, inducing_y)#, diag_reg=1e-4)
#     _, B_B = predict_fn(x_test=x_batch, get="nngp", compute_cov=True)

#     kernel = kernel_fn(batch_induced, batch_induced, "nngp")
#     k_b_b, k_b_i, k_i_b, k_i_i = split_kernel(kernel, batch_num)
#     k_i_i_inverse = jnp.linalg.inv(k_i_i + 1e-0 * jnp.eye(induce_num))

#     A_B = matmul2(k_b_i, k_i_i_inverse)
#     L = jax.scipy.linalg.cholesky(k_i_i, lower=True)
#     A_B_L = matmul2(A_B, L)
#     A_B_L = kron_diag(A_B_L, n=class_num)
#     B_B = kron_diag(B_B, n=class_num)

#     mean = matmul2(A_B_L, inducing_mu)
#     covariance = matmul3(A_B_L, inducing_sigma_mat, A_B_L.T) + B_B
#     covariance_L = jax.scipy.linalg.cholesky(covariance, lower=True)

#     return mean, covariance_L

def mean_covariance(
    x_batch, inducing_points, kernel_fn, l,
    inducing_mu, inducing_sigma_mat,
    batch_num, induce_num, class_num, kernel_scale
):
    batch_induced = jnp.concatenate([x_batch, inducing_points], axis=0)

    kernel = kernel_fn(batch_induced, l)

    # inducing_x = inducing_points
    # inducing_y = jnp.zeros((induce_num, class_num))
    # predict_fn = gradient_descent_mse_ensemble(kernel_fn, inducing_x, inducing_y)#, diag_reg=1e-4)
    # _, B_B = predict_fn(x_test=x_batch, get="nngp", compute_cov=True)

    B_B *= kernel_scale

    kernel = kernel_fn(batch_induced, batch_induced, "nngp") * kernel_scale

    k_b_b, k_b_i, k_i_b, k_i_i = split_kernel(kernel, batch_num)
    k_i_i_inverse = jnp.linalg.inv(k_i_i + 1e-4 * jnp.eye(induce_num))

    B_B = k_b_b - matmul3(k_b_i, k_i_i_inverse, k_i_b)
    A_B = matmul2(k_b_i, k_i_i_inverse)

    # L = jnp.linalg.cholesky(k_i_i)
    # A_B_L = matmul2(A_B, L)
    A_B_L = A_B
    A_B_L = kron_diag(A_B_L, n=class_num)
    B_B = kron_diag(B_B, n=class_num)

    mean = matmul2(A_B_L, inducing_mu)
    covariance = matmul3(A_B_L, inducing_sigma_mat, A_B_L.T) + B_B
    covariance_L = jax.scipy.linalg.cholesky(covariance, lower=True)

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


def g_kl_divergence(inducing_mu, inducing_sigma_mat, class_num, induce_num, inducing_points, kernel_fn, kernel_scale):

    k_i_i = kernel_fn(inducing_points, inducing_points, "nngp") * kernel_scale
    k_i_i_inverse = jnp.linalg.inv(k_i_i + 1e-4 * jnp.eye(induce_num))

    s_sigma, alogdetsigma = jnp.linalg.slogdet(k_i_i)
    logdet_k_i_i = s_sigma * alogdetsigma

    s_sigma, alogdetsigma = jnp.linalg.slogdet(inducing_sigma_mat)
    logdet_sigma = s_sigma * alogdetsigma

    k_i_i_inverse = kron_diag(k_i_i_inverse, n=class_num)

    kl = 1 / 2 * (logdet_k_i_i - logdet_sigma - induce_num * class_num \
       + jnp.trace(jnp.matmul(k_i_i_inverse, inducing_sigma_mat)) + matmul3(inducing_mu[None, ...], k_i_i_inverse, inducing_mu[..., None]))
    return jnp.sum(kl)


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

    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_values(sampled_f_softmax, label)
    ll = jnp.sum(logsumexp(true_label_softmax.T) - jnp.log(sample_num))

    return ll


def get_correct_count(label, sampled_f, class_num, sample_num, test_num):
    sampled_f = jnp.transpose(sampled_f.reshape(sample_num, class_num, test_num), axes=(0, 2, 1))

    sampled_f_softmax = jnp.exp(log_softmax(sampled_f))
    mean_f_softmax = jnp.mean(sampled_f_softmax, axis=0)
    predict_y = jnp.argmax(mean_f_softmax, axis=-1)
    true_y = jnp.argmax(label, axis=-1)
    correct_count = jnp.sum(predict_y == true_y)

    return correct_count


def g_ELBO(
    x_batch, y_batch, kernel_fn,
    inducing_points, inducing_mu, inducing_sigma,
    kernel_scale, nums, seed=0,
):
    sample_num = nums.sample_num
    batch_num = nums.batch_num
    induce_num = nums.induce_num
    train_num = nums.train_num
    class_num = nums.class_num

    inducing_sigma_mat = jnp.diag(inducing_sigma)

    mean, covariance_L = mean_covariance(x_batch, inducing_points, kernel_fn, l,
                                         inducing_mu, inducing_sigma_mat,
                                         batch_num, induce_num, class_num, kernel_scale)

    # key = random.PRNGKey(seed)
    # sampled_f = g_sample_f_b(sample_num, batch_num, class_num, key)
    # sampled_f = (mean.reshape(-1, 1) + matmul2(covariance_L, sampled_f.T)).T

    ll = log_likelihood(y_batch, sampled_f, batch_num, class_num, sample_num, train_num)
    kl = g_kl_divergence(inducing_mu, inducing_sigma_mat, class_num, induce_num, inducing_points, kernel_fn, kernel_scale)

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


def g_test(x_test, y_test, kernel_fn, inducing_points, inducing_mu, inducing_sigma, key, nums):
    test_num = nums.test_num
    induce_num = nums.induce_num
    class_num = nums.class_num
    test_sample_num = nums.test_sample_num

    inducing_sigma = jnp.diag(inducing_sigma)

    test_nll_list = []
    total_correct_count = 0

    test_batches = TestBatch(x_test, y_test, 64)
    for batch_x, batch_y in tqdm(test_batches):
        batch_num = batch_x.shape[0]

        induced_test = jnp.concatenate([inducing_points, batch_x], axis=0)
        kernel = kernel_fn(induced_test, induced_test, "nngp")

        kernel_i_i, kernel_i_t, kernel_t_i, kernel_t_t = split_kernel(kernel, induce_num)
        kernel_i_i_inverse = jnp.linalg.inv(kernel_i_i + 1e-4 * jnp.eye(induce_num))

        # L_induced = jnp.linalg.cholesky(kernel_i_i)

        # A_L = matmul3(kernel_t_i, kernel_i_i_inverse, L_induced)
        A_L = jnp.matmul(kernel_t_i, kernel_i_i_inverse)
        A_L = kron_diag(A_L, n=class_num)

        # L_induced = kron_diag(L_induced, n=class_num)
        # L_mu = matmul2(L_induced, inducing_mu)

        inducing_x = inducing_points
        # inducing_y = jnp.transpose(L_mu.reshape(-1, induce_num))
        inducing_y = jnp.transpose(inducing_mu.reshape(-1, induce_num))
        predict_fn = gradient_descent_mse_ensemble(kernel_fn, inducing_x, inducing_y, diag_reg=1e-4)

        test_mean, test_covariance = predict_fn(x_test=batch_x, get="nngp", compute_cov=True)
        test_mean = test_mean.T.flatten()
        test_covariance = kron_diag(test_covariance, n=class_num)

        A_sigma = matmul3(A_L, inducing_sigma, A_L.T)
        test_sigma = A_sigma + test_covariance

        _, key_test = random.split(key)

        test_f_sample = random.multivariate_normal(
            key=key_test,
            mean=test_mean,
            cov=test_sigma,
            shape=(test_sample_num,),
        )

        batch_test_nll = -log_likelihood2(batch_y, test_f_sample, class_num, test_sample_num, batch_num)
        test_nll_list.append(batch_test_nll)

        batch_correct_count = get_correct_count(batch_y, test_f_sample, class_num, test_sample_num, batch_num)
        total_correct_count += batch_correct_count

    nll = jnp.sum(jnp.array(test_nll_list)) / test_num
    acc = total_correct_count / test_num

    return nll, acc


def g_main(dataset, log_name, train_num, test_num, batch_num, induce_num, sample_num,
         test_sample_num, learning_rate, epochs, print_interval, inducing, kernel_scale, **kwargs):

    seed = 10

    if log_name is not None:
        log_file = open(log_name, "w")

    depth = 3

    # Dataset
    if dataset == "mnist":
        train_num   = 10000
        test_num    = 1000
    elif dataset == "iris":
        train_num   = 120
        test_num    = 30
    elif dataset == "test":
        train_num   = 8500
        test_num    = 150

    normalize       = True
    batch_num       = 128
    induce_num      = 100
    sample_num      = 1000
    # test_sample_num = 100
    optimizer       = "adam"
    learning_rate   = 0.0001
    # epochs          = 13000
    print_interval  = 100
    depth           = 4

    print(f"{train_num}, {kernel_scale}, {inducing}, {normalize}")

    x_train, y_train, x_test, y_test = get_dataset(
        dataset, train_num, test_num, normalize=normalize, seed=seed
    )
    class_num = y_train.shape[1]

    # Kernel
    if dataset == "mnist":
        kernel_fn = get_cnn_kernel(depth=depth, class_num=class_num)
    elif dataset == "iris" or dataset == "test":
        # kernel_fn = get_mlp_kernel(depth=depth, class_num=class_num, W_std=W_std, b_std=b_std)
        kernel_fn = rbf

    # kernel = kernel_fn(x_train,x_train, 'nngp')


    # Init
    key = random.PRNGKey(10)
    # inducing_points = random.normal(key, shape=(induce_num, *x_train.shape[1:]))

    inducing_points = x_train[:induce_num]
    inducing_mu = jnp.zeros(induce_num * class_num)
    inducing_sigma = jnp.ones(induce_num * class_num)

    if inducing:
        # train_params = (inducing_points, inducing_mu, inducing_sigma, kernel_scale)
        train_params = (inducing_points, inducing_mu, inducing_sigma)
    else:
        # train_params = (inducing_mu, inducing_sigma, kernel_scale)
        train_params = (inducing_mu, inducing_sigma)

    # TODO: Refactoring
    nums = namedtuple("nums", [
        "train_num", "test_num", "batch_num", "induce_num",
        "sample_num", "test_sample_num", "class_num",
    ])(train_num, test_num, batch_num, induce_num, sample_num, test_sample_num, class_num)

    # Grads

    # ELBO_jit = g_ELBO
    ELBO_jit = jit(g_ELBO, static_argnums=(2, 6, 7, 8))
    if inducing:
        # grad_elbo = grad(ELBO_jit, argnums=(3, 4, 5, 6))
        grad_elbo = grad(ELBO_jit, argnums=(3, 4, 5))
    else:
        # grad_elbo = grad(ELBO_jit, argnums=(4, 5, 6))
        grad_elbo = grad(ELBO_jit, argnums=(4, 5))
    grad_elbo = jit(grad_elbo, static_argnums=(2, 6, 7, 8))

    if optimizer == "adam":
        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    elif optimizer == "sgd":
        opt_init, opt_update, get_params = optimizers.sgd(learning_rate)

    opt_state = opt_init(train_params)

    # Training
    train_batches = TrainBatch(x_train, y_train, batch_size=batch_num, epochs=epochs+1, seed=seed)

    for i, (x_batch, y_batch) in tqdm(enumerate(train_batches), total=epochs):
        train_params = get_params(opt_state)

        if i % print_interval == 0:
            if inducing:
                elbo = ELBO_jit(x_batch, y_batch, kernel_fn, *train_params, kernel_scale, nums)
            else:
                elbo = ELBO_jit(x_batch, y_batch, kernel_fn, inducing_points, *train_params, kernel_scale, nums)

            tqdm.write("{} / {}: ELBO = {:.7f}".format(i, len(train_batches) - 1, elbo))
            if log_name is not None:
                log_file.write("{} / {}: ELBO = {:.7f}\n".format(i, len(train_batches) - 1, elbo))
                log_file.flush()

        if inducing:
            opt_state = opt_update(i, grad_elbo(x_batch, y_batch, kernel_fn, *train_params, kernel_scale, nums), opt_state)
        else:
            opt_state = opt_update(i, grad_elbo(x_batch, y_batch, kernel_fn, inducing_points, *train_params, kernel_scale, nums), opt_state)



    # Test
    if inducing:
        inducing_points, inducing_mu, inducing_sigma = get_params(opt_state)
        # np.save("test/inducing_points.npy", inducing_points)
        # np.save("test/inducing_mu.npy", inducing_mu)
        # np.save("test/inducing_sigma.npy", inducing_sigma)
    else:
        inducing_mu, inducing_sigma = get_params(opt_state)
        # np.save("test/inducing_points.npy", inducing_points)
        # np.save("test/inducing_sigma.npy", inducing_sigma)

    test_nll, test_acc = g_test(x_test, y_test, kernel_fn, inducing_points, inducing_mu, inducing_sigma, key, nums)

    print(test_nll, test_acc)

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
