import numpy as np

from jax import jit
from jax import numpy as jnp
from jax.numpy import linalg
from scipy import stats
from neural_tangents.predict import gradient_descent_mse_ensemble
from neural_tangents import stax

from . import data
from .ops import logsumexp


def add_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("importance-sampling",
                                   aliases=["is"],
                                   help="importance-sampling",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=main)

    parser.add_argument("-d", "--dataset", choices=data.regression_datasets)
    parser.add_argument("-ds", "--dist", default="burr12")
    parser.add_argument("-sn", "--sample-num", default=1000, type=int)
    parser.add_argument("-nh", "--num-hiddens", default=1, type=int)
    parser.add_argument("-wv", "--w-variance", default=1., type=float)
    parser.add_argument("-bv", "--b-variance", default=0., type=float)
    parser.add_argument("-act", "--activation", default="erf", choices=["erf", "relu"])
    parser.add_argument("-a", "--alpha", default=2., type=float)
    parser.add_argument("-b", "--beta", default=2., type=float)
    parser.add_argument("-e", "--epsilon-log-variance", default=4., type=float)
    parser.add_argument("-lv", "--last-layer-variance", default=1., type=float)
    parser.add_argument("-s", "--seed", default=10, type=int)
    parser.add_argument("-bc", "--burr-c", default=2., type=float)
    parser.add_argument("-bd", "--burr-d", default=1., type=float)

    parser.add_argument("-np", "--nu-p", default=4., type=float)
    parser.add_argument("-rp", "--rho-p", default=4., type=float)
    parser.add_argument("-nq", "--nu-q", default=4., type=float)
    parser.add_argument("-rq", "--rho-q", default=4., type=float)


def get_kernel_fn(depth, W_std, b_std, last_W_std=1., act="erf"):
    layers = []

    for _ in range(depth):
        layers.append(stax.Dense(8192, W_std=W_std, b_std=b_std))

        if act == "erf":
            layers.append(stax.Erf())
        elif act == "relu":
            layers.append(stax.Relu())

    layers.append(stax.Dense(1, W_std=last_W_std))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    return kernel_fn


def main(dataset, num_hiddens, w_variance, b_variance, activation,
         epsilon_log_variance, last_layer_variance, seed, dist,
         burr_c, burr_d, nu_p, rho_p, nu_q, rho_q, sample_num, **kwargs):

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

    train_valid_x = np.concatenate([train_x, valid_x], axis=0)
    train_valid_y = np.concatenate([train_y, valid_y], axis=0)

    train_valid_x, train_valid_y = data.permute_dataset(train_valid_x, train_valid_y, seed=seed)
    train_x, valid_x = train_valid_x[:train_num], train_valid_x[train_num:]
    train_y, valid_y = train_valid_y[:train_num], train_valid_y[train_num:]

    # Models
    W_std = jnp.sqrt(w_variance)
    b_std = jnp.sqrt(b_variance)
    last_W_std = jnp.sqrt(last_layer_variance)

    kernel_fn_is = get_kernel_fn(num_hiddens, W_std=W_std, b_std=b_std, last_W_std=1.)
    kernel_fn_const = get_kernel_fn(num_hiddens, W_std=W_std, b_std=b_std, last_W_std=last_W_std)

    # Importance

    nngp_train_covariance = kernel_fn_is(train_x, train_x, 'nngp')  # K_NNGP(X_D, X_D)

    if dist == "invgamma":
        sample_q = stats.invgamma.rvs(a=(nu_q/2), scale=(rho_q/2), size=sample_num, random_state=101)
    elif dist == "burr12":
        sample_q = stats.burr12.rvs(c=burr_c, d=burr_d, loc=0., scale=1., size=sample_num, random_state=101)
    else:
        raise ValueError

    minus_log_two_pi = -(train_num / 2) * jnp.log(2 * jnp.pi)
    minus_y_train_K_NNGP_y_train = -(1 / 2) * jnp.matmul(jnp.matmul(train_y.T, linalg.inv(nngp_train_covariance + 1e-4 * jnp.eye(train_num))), train_y)
    minus_log_det_K_NNGP = stats.multivariate_normal.logpdf(train_y.reshape(train_num), None, nngp_train_covariance + 1e-4 * jnp.eye(train_num)) - minus_log_two_pi - minus_y_train_K_NNGP_y_train
    minus_log_sigma = -(1 / 2) * train_num * jnp.log(sample_q)

    log_probability_data = minus_log_two_pi + minus_log_det_K_NNGP + minus_y_train_K_NNGP_y_train / sample_q + minus_log_sigma
    log_probability_data = log_probability_data - log_probability_data.max()
    probability_data = jnp.exp(log_probability_data).reshape(-1, 1)

    if dist == "invgamma":
        probability_prior = stats.invgamma.pdf(sample_q, a=(nu_p/2), scale=(rho_p/2)).reshape(-1, 1)#[:, None]
        probability_q  = stats.invgamma.pdf(sample_q, a=(nu_p/2), scale=(rho_p/2)).reshape(-1, 1)#[:, None]
    elif dist == "burr12":
        probability_prior = stats.burr12.pdf(sample_q, c=burr_c, d=burr_d, loc=0., scale=1.).reshape(-1, 1)#[:, None]
        probability_q  = stats.burr12.pdf(sample_q, c=burr_c, d=burr_d, loc=0., scale=1.).reshape(-1, 1)#[:, None]
    else:
        raise ValueError

    probability_joint = probability_data * probability_prior
    w = probability_joint / probability_q
    w_bar = w / jnp.sum(w)

    predict_fn = gradient_descent_mse_ensemble(kernel_fn_is, train_x, train_y, diag_reg=epsilon_variance)
    nngp_mean_is_valid, nngp_covariance_is_valid = predict_fn(x_test=valid_x, get="nngp", compute_cov=True)
    nngp_mean_is_test, nngp_covariance_is_test = predict_fn(x_test=test_x, get="nngp", compute_cov=True)

    nngp_std_diag_is_valid = jnp.sqrt(jnp.diag(nngp_covariance_is_valid))
    nngp_std_is_valid = jnp.sqrt(sample_q[:, None, None]) * nngp_std_diag_is_valid[None, :, None]
    nngp_mean_is_valid = jnp.tile(nngp_mean_is_valid[None, :, None], (sample_num, 1, 1))

    nngp_std_diag_is_test = jnp.sqrt(jnp.diag(nngp_covariance_is_test))
    nngp_std_is_test = jnp.sqrt(sample_q[:, None, None]) * nngp_std_diag_is_test[None, :, None]
    nngp_mean_is_test = jnp.tile(nngp_mean_is_test[None, :, None], (sample_num, 1, 1))

    # Const

    predict_fn_const = gradient_descent_mse_ensemble(kernel_fn_const, train_x, train_y, diag_reg=epsilon_variance)
    nngp_mean_const_valid, nngp_covariance_const_valid = predict_fn_const(x_test=valid_x, get="nngp", compute_cov=True)
    nngp_mean_const_test, nngp_covariance_const_test = predict_fn_const(x_test=test_x, get="nngp", compute_cov=True)

    # nngp_mean_const_valid = nngp_mean_const_valid[:, None, None]
    nngp_std_const_valid = jnp.sqrt(jnp.diag(nngp_covariance_const_valid))

    # nngp_mean_const_test = nngp_mean_const_test[:, None, None]
    nngp_std_const_test = jnp.sqrt(jnp.diag(nngp_covariance_const_test))


    def mcnll_const(test_y, mean, std):
        neg_log_prob = -stats.norm.logpdf(test_y, mean, std)
        return neg_log_prob

    def mcnll_is(test_y, mean, std, w_bar):
        test_y = jnp.concatenate([test_y] * len(mean)).reshape(-1,1)
        prob = jnp.sum(w_bar * stats.norm.pdf(test_y, mean, std))
        neg_log_prob = -jnp.log(prob)
        return neg_log_prob

    # def mcnll_is(test_y, mean, std, w_bar):
    #     test_y = jnp.concatenate([test_y] * len(mean)).reshape(-1,1)
    #     neg_log_prob = -logsumexp(jnp.log(w_bar) + stats.norm.logpdf(test_y, mean, std))
    #     return jnp.sum(neg_log_prob)

    # calculate nll for test points
    neg_log_prob_is_valid = 0
    for i, valid_y_i in enumerate(valid_y):
        neg_log_prob_is_valid += mcnll_is(valid_y_i, nngp_mean_is_valid[:, i], nngp_std_is_valid[:, i], w_bar)
    neg_log_prob_is_valid /= valid_num

    neg_log_prob_is_test = 0
    for i, test_y_i in enumerate(test_y):
        neg_log_prob_is_test += mcnll_is(test_y_i, nngp_mean_is_test[:, i], nngp_std_is_test[:, i], w_bar)
    neg_log_prob_is_test /= test_num

    neg_log_prob_constant_valid = 0
    for i, valid_y_i in enumerate(valid_y):
        neg_log_prob_constant_valid += jnp.squeeze(mcnll_const(valid_y_i, nngp_mean_const_valid[i], nngp_std_const_valid[i]))
    neg_log_prob_constant_valid /= valid_num

    neg_log_prob_constant_test = 0
    for i, test_y_i in enumerate(test_y):
        neg_log_prob_constant_test += jnp.squeeze(mcnll_const(test_y_i, nngp_mean_const_test[i], nngp_std_const_test[i]))
    neg_log_prob_constant_test /= test_num


    print("------------------------------------------------------------------")
    print("num_hiddens: {:<2d}  / act:    {}".format(num_hiddens, activation))
    print("w_variance:  {:1.1f} / burr-c: {:1.1f}".format(w_variance, burr_c))
    print("b_variance:  {:1.1f} / burr-d: {:1.1f}".format(b_variance, burr_d))
    print("epsilon_log_variance: {}".format(raw_epsilon_log_variance))
    print("last_layer_variance: {}     / seed: {}".format(last_layer_variance, seed))
    print("---------------------------------------------")
    print("Valid NLL for burr 12 prior:  [{:13.8f}]".format(neg_log_prob_is_valid))
    print("Valid NLL for constant prior: [{:13.8f}]".format(neg_log_prob_constant_valid))
    print("Test NLL for burr 12 prior:   [{:13.8f}]".format(neg_log_prob_is_test))
    print("Test NLL for constant prior:  [{:13.8f}]".format(neg_log_prob_constant_test))
