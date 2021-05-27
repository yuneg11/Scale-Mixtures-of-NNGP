import numpy as np

from jax import jit
from jax import numpy as jnp
from scipy import stats
import neural_tangents as nt
from neural_tangents import stax

from . import data
from . import utils


def add_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("regression",
                                   aliases=["reg"],
                                   help="regression",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=main)

    parser.add_argument("-d", "--dataset", choices=data.regression_datasets)
    parser.add_argument("-nh", "--num-hiddens", default=1, type=int)
    parser.add_argument("-wv", "--w-variance", default=1., type=float)
    parser.add_argument("-bv", "--b-variance", default=0., type=float)
    parser.add_argument("-act", "--activation", default="erf", choices=["erf", "relu"])
    parser.add_argument("-a", "--alpha", default=2., type=float)
    parser.add_argument("-b", "--beta", default=2., type=float)
    parser.add_argument("-e", "--epsilon-log-variance", default=8., type=float)
    parser.add_argument("-lv", "--last-layer-variance", default=1., type=float)
    parser.add_argument("-s", "--seed", default=10, type=int)


def main(dataset, num_hiddens, w_variance, b_variance, activation,
         alpha, beta, epsilon_log_variance, last_layer_variance, seed, **kwargs):

    raw_epsilon_log_variance = epsilon_log_variance
    epsilon_log_variance = -6 + epsilon_log_variance / 2
    epsilon_variance = jnp.power(10, epsilon_log_variance)
    # last_layer_variance = 2 * beta / (2 * alpha - 2)

    # dataset
    x, y = data.get_dataset(dataset, "./data", y_newaxis=True)
    x, y = data.permute_dataset(x, y, seed=10)
    splits = data.split_dataset(x, y, train=0.8, valid=0.1, test=0.1)
    train_x, train_y, valid_x, valid_y, test_x, test_y = splits

    train_num = len(train_x)
    valid_num = len(valid_x)
    test_num = len(test_x)

    if seed >= 0:
        train_valid_x = np.concatenate([train_x, valid_x], axis=0)
        train_valid_y = np.concatenate([train_y, valid_y], axis=0)

        train_valid_x, train_valid_y = data.permute_dataset(train_valid_x, train_valid_y, seed=seed)
        train_x, valid_x = train_valid_x[:train_num], train_valid_x[train_num:]
        train_y, valid_y = train_valid_y[:train_num], train_valid_y[train_num:]

    train_valid = jnp.concatenate([train_x, valid_x], axis=0)
    train_test = jnp.concatenate([train_x, test_x], axis=0)

    # Models

    base_layers = []

    for _ in range(num_hiddens):
        base_layers.append(stax.Dense(8192, W_std=jnp.sqrt(w_variance), b_std=jnp.sqrt(b_variance)))

        if activation == "erf":
            base_layers.append(stax.Erf())
        elif activation == "relu":
            base_layers.append(stax.Relu())

    layers1 = base_layers + [stax.Dense(1, W_std=jnp.sqrt(last_layer_variance), b_std=0.)]
    init_fn_1, apply_fn_1, kernel_fn_1 = stax.serial(*layers1)
    kernel_fn_1 = jit(kernel_fn_1, static_argnums=(2,))

    layers2 = base_layers + [stax.Dense(1, W_std=1, b_std=0.)]
    init_fn_2, apply_fn_2, kernel_fn_2 = stax.serial(*layers2)
    kernel_fn_2 = jit(kernel_fn_2, static_argnums=(2,))

    predict_fn_1 = nt.predict.gradient_descent_mse_ensemble(kernel_fn_1, train_x, train_y, diag_reg=epsilon_variance)
    nngp_mean_valid, nngp_covariance_valid = predict_fn_1(x_test=valid_x, get="nngp", compute_cov=True)
    nngp_mean_test, nngp_covariance_test = predict_fn_1(x_test=test_x, get="nngp", compute_cov=True)

    nngp_mean_valid = nngp_mean_valid.flatten()
    nngp_std_valid = jnp.sqrt(jnp.diag(nngp_covariance_valid))
    nngp_mean_test = nngp_mean_test.flatten()
    nngp_std_test = jnp.sqrt(jnp.diag(nngp_covariance_test))

    predict_fn_2 = nt.predict.gradient_descent_mse_ensemble(kernel_fn_2, train_x, train_y, diag_reg=epsilon_variance)
    _, nngp_covariance_2_valid = predict_fn_2(x_test=valid_x, get="nngp", compute_cov=True)
    _, nngp_covariance_2_test = predict_fn_2(x_test=test_x, get="nngp", compute_cov=True)

    # Valid

    train_valid_nngp_kernel = kernel_fn_2(train_valid, train_valid, "nngp")

    nu = 2 * alpha
    mean = jnp.zeros(train_num + valid_num)
    covariance = beta / alpha * train_valid_nngp_kernel

    conditional_nu = nu + train_num

    split_kernels = utils.split_kernel(covariance, train_num)
    kernel_train_train, kernel_train_valid, kernel_valid_train, kernel_valid_valid = split_kernels

    inverse_k_11 = jnp.linalg.inv(kernel_train_train + epsilon_variance * jnp.eye(train_num))
    d_1 = nu + jnp.matmul(jnp.matmul(train_y.T, inverse_k_11),train_y)

    posterior_kernel = nngp_covariance_2_valid
    conditional_kernel = d_1 / conditional_nu * beta / alpha * posterior_kernel

    valid_std = jnp.sqrt(jnp.diag(conditional_kernel))

    # Test

    train_test_nngp_kernel = kernel_fn_2(train_test, train_test, "nngp")

    nu = 2 * alpha
    mean = jnp.zeros(train_num + test_num)
    covariance = beta / alpha * train_test_nngp_kernel

    split_kernels = utils.split_kernel(covariance, train_num)
    kernel_train_train, kernel_train_test, kernel_test_train, kernel_test_test = split_kernels

    posterior_kernel = nngp_covariance_2_test
    conditional_kernel = d_1 / conditional_nu * beta / alpha * posterior_kernel

    test_std = jnp.sqrt(jnp.diag(conditional_kernel))

    # make nll calculator for test points

    def gaussian_nll(test_y, mean, std):
        neg_log_prob = -stats.norm.logpdf(test_y, mean, std)
        return neg_log_prob

    def studentt_nll(test_y, freedom, mean, std):
        neg_log_prob = -stats.t.logpdf(test_y, freedom, mean, std)
        return neg_log_prob


    valid_neg_log_prob_invgamma = jnp.mean(jnp.array([
        studentt_nll(y, conditional_nu, nngp_mean, std)
        for y, nngp_mean, std
        in zip(valid_y, nngp_mean_valid, valid_std)
    ]))

    test_neg_log_prob_invgamma = jnp.mean(jnp.array([
        studentt_nll(y, conditional_nu, nngp_mean, std)
        for y, nngp_mean, std
        in zip(test_y, nngp_mean_test, test_std)
    ]))

    valid_neg_log_prob_constant = jnp.mean(jnp.array([
        gaussian_nll(y, nngp_mean, nngp_std)
        for y, nngp_mean, nngp_std
        in zip(valid_y, nngp_mean_valid, nngp_std_valid)
    ]))

    test_neg_log_prob_constant = jnp.mean(jnp.array([
        gaussian_nll(y, nngp_mean, nngp_std)
        for y, nngp_mean, nngp_std
        in zip(test_y, nngp_mean_test, nngp_std_test)
    ]))

    print("-------------------------------------------------------------")
    print("num_hiddens: {:<2d}  / act:   {}".format(num_hiddens, activation))
    print("w_variance:  {:1.1f} / alpha: {:1.1f}".format(w_variance, alpha))
    print("b_variance:  {:1.1f} / beta:  {:1.1f}".format(b_variance, beta))
    print("epsilon_log_variance: {:.2f}".format(raw_epsilon_log_variance))
    print("last_layer_variance: {:.2f}        / seed: {}".format(last_layer_variance, seed))
    print("---------------------------------------------")
    print("Valid NLL for invgamma prior: [{:13.8f}]".format(valid_neg_log_prob_invgamma))
    print("Valid NLL for constant prior: [{:13.8f}]".format(valid_neg_log_prob_constant))
    print("Test NLL for invgamma prior:  [{:13.8f}]".format(test_neg_log_prob_invgamma))
    print("Test NLL for constant prior:  [{:13.8f}]".format(test_neg_log_prob_constant))
