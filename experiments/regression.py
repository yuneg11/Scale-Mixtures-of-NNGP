import numpy as np
from scipy import stats

from jax import jit
from jax import numpy as jnp

from neural_tangents import stax
from neural_tangents.predict import gradient_descent_mse_ensemble

from . import data
from . import utils
from .ops import *


def add_subparser(subparsers):
    parser = subparsers.add_parser("regression", aliases=["reg"], help="regression")
    parser.set_defaults(func=main)

    parser.add_argument("dataset", choices=data.regression_datasets)
    parser.add_argument("-nh", "--num-hiddens", default=1, type=int)
    parser.add_argument("-wv", "--w-variance", default=1., type=float)
    parser.add_argument("-bv", "--b-variance", default=0., type=float)
    parser.add_argument("-act", "--activation", default="erf", choices=["erf", "relu"])
    parser.add_argument("-e", "--epsilon-log-variance", default=5., type=float)
    parser.add_argument("-s", "--seed", default=10, type=int)

    parser.add_argument("-lv", "--last-layer-variance")
    
    parser.add_argument("-a", "--alpha")
    parser.add_argument("-b", "--beta")

    parser.add_argument("-bc", "--burr-c")
    parser.add_argument("-bd", "--burr-d")
    parser.add_argument("-sn", "--sample-num", default=1000, type=int)


def get_kernel_fn(depth, W_std, b_std, last_W_std=1., act="erf"):
    if act == "relu":
        act_class = stax.Relu
    elif act == "erf":
        act_class = stax.Erf
    else:
        raise KeyError("Unsupported act '{}'".format(act))

    layers = []
    for _ in range(depth):
        layers.append(stax.Dense(8192, W_std=W_std, b_std=b_std))
        layers.append(act_class())

    layers.append(stax.Dense(1, W_std=last_W_std))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    return kernel_fn


def gaussian_nll(y_test, mean, std):
    neg_log_prob = -stats.norm.logpdf(y_test.flatten(), mean.flatten(), std)
    return neg_log_prob


def student_t_nll(y_test, df, mean, std):
    neg_log_prob = -stats.t.logpdf(y_test.flatten(), df, mean.flatten(), std)
    return neg_log_prob


def mc_nll(y_test, mean, std, w_bar):
    y_test = concatenate([y_test] * len(mean)).reshape(-1, 1)
    log_probs = log(w_bar + 1e-24) + stats.norm.logpdf(y_test, mean, std)
    neg_log_prob = -logsumexp(log_probs.flatten(), axis=0)
    return sum(neg_log_prob)


def main(dataset, num_hiddens, w_variance, b_variance, activation, burr_c, burr_d, alpha, beta,
         epsilon_log_variance, seed, sample_num, last_layer_variance, **kwargs):    
    
    # Argument process

    base_args = {
        "num-hiddens": num_hiddens,
        "w-variance": w_variance,
        "b-variance": b_variance,
        "activation": activation,
        "eps-log-var": epsilon_log_variance,
        "seed": seed,
    }

    W_std = sqrt(w_variance)
    b_std = sqrt(b_variance)
    eps = power(10, -6 + epsilon_log_variance / 2)

    invgamma, alpha_set, beta_set = utils.process_invgamma_args(alpha, beta)
    burr12, burr_c_set, burr_d_set = utils.process_burr12_args(burr_c, burr_d)
    const, last_W_std = utils.process_const_args(last_layer_variance)

    if not const and not invgamma and not burr12:
        print("No distribution is selected.")
        exit(-1)

    if not const and not invgamma and not burr12:
        print("No distribution is selected.")
        exit(-1)

    # Dataset

    x, y = data.get_dataset(dataset, "./data", y_newaxis=True)
    x, y = data.permute_dataset(x, y, seed=10)
    splits = data.split_dataset(x, y, train=0.8, valid=0.1, test=0.1)
    x_train, y_train, x_valid, y_valid, x_test, y_test = splits

    train_num = len(x_train)
    valid_num = len(x_valid)
    test_num = len(x_test)

    if seed >= 0:
        train_valid_x = np.concatenate([x_train, x_valid], axis=0)
        train_valid_y = np.concatenate([y_train, y_valid], axis=0)

        train_valid_x, train_valid_y = data.permute_dataset(train_valid_x, train_valid_y, seed=seed)
        x_train, x_valid = train_valid_x[:train_num], train_valid_x[train_num:]
        y_train, y_valid = train_valid_y[:train_num], train_valid_y[train_num:]

    # Compute

    if invgamma or burr12:
        kernel_fn = get_kernel_fn(num_hiddens, W_std=W_std, b_std=b_std, last_W_std=1., act=activation)
        predict_fn = gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=eps)
        mean_valid, cov_valid = predict_fn(x_test=x_valid, get="nngp", compute_cov=True)
        mean_test, cov_test = predict_fn(x_test=x_test, get="nngp", compute_cov=True)
        cov_train = kernel_fn(x_train, x_train, get="nngp")

    if const:
        if (invgamma or burr12) and (-1e-10 <= last_W_std - 1. <= 1e-10):
            mean_valid_const, cov_valid_const = mean_valid, cov_valid
            mean_test_const, cov_test_const = mean_test, cov_test
        else:
            kernel_fn_const = get_kernel_fn(num_hiddens, W_std=W_std, b_std=b_std,
                                            last_W_std=last_W_std, act=activation)
            predict_fn_const = gradient_descent_mse_ensemble(kernel_fn_const, x_train, y_train, diag_reg=eps)
            mean_valid_const, cov_valid_const = predict_fn_const(x_test=x_valid, get="nngp", compute_cov=True)
            mean_test_const, cov_test_const = predict_fn_const(x_test=x_test, get="nngp", compute_cov=True)

        std_diag_valid_const = sqrt(diag(cov_valid_const))
        std_diag_test_const = sqrt(diag(cov_test_const))

        nll_valid_const = mean(gaussian_nll(y_valid, mean_valid_const, std_diag_valid_const))
        nll_test_const = mean(gaussian_nll(y_test, mean_test_const, std_diag_test_const))

        utils.print_yaml(type="const", **base_args,
                         **{"last-layer-var": last_layer_variance,
                            "valid-nll": round(float(nll_valid_const), 6),
                            "test-nll": round(float(nll_test_const), 6)})

    if invgamma:
        for alpha in alpha_set:
            for beta in beta_set:
                nu = 2 * alpha
                cond_nu = nu + train_num

                inv_cov_train = inv(beta / alpha * cov_train + eps * eye(train_num))
                d_1 = nu + matmul3(y_train.T, inv_cov_train, y_train).flatten()
                
                std_valid = sqrt(diag(d_1 / cond_nu * beta / alpha * cov_valid))
                std_test = sqrt(diag(d_1 / cond_nu * beta / alpha * cov_test))

                nll_valid_invgamma = mean(student_t_nll(y_valid, cond_nu, mean_valid, std_valid))
                nll_test_invgamma = mean(student_t_nll(y_test, cond_nu, mean_test, std_test))

                utils.print_yaml(type="invgamma", **base_args,
                                **{"alpha": alpha, "beta": beta,
                                   "valid-nll": round(float(nll_valid_invgamma), 6),
                                   "test-nll": round(float(nll_test_invgamma), 6)})

    if burr12:
        minus_log_two_pi = -(train_num / 2) * log(2 * np.pi)
        minus_y_train_K_NNGP_y_train = -(1 / 2) * matmul3(y_train.T, inv(cov_train + 1e-4 * eye(train_num)), y_train)
        normal_x = y_train.reshape(train_num)
        normal_cov = cov_train + 1e-4 * eye(train_num)
        minus_log_det_K_NNGP = stats.multivariate_normal.logpdf(normal_x, None, normal_cov, allow_singular=True) \
                             - minus_log_two_pi - minus_y_train_K_NNGP_y_train

        std_diag_valid = sqrt(diag(cov_valid))
        mean_valid = jnp.tile(mean_valid[None, :, None], (sample_num, 1, 1))
        std_diag_test = sqrt(diag(cov_test))
        mean_test = jnp.tile(mean_test[None, :, None], (sample_num, 1, 1))

        for burr_c in burr_c_set:
            for burr_d in burr_d_set:
                sample_q = stats.burr12.rvs(c=burr_c, d=burr_d, loc=0., scale=1., size=sample_num, random_state=101)
                minus_log_sigma = -(1 / 2) * train_num * log(sample_q)

                log_prob_data = minus_log_two_pi + minus_log_det_K_NNGP + minus_y_train_K_NNGP_y_train / sample_q + minus_log_sigma
                prob_data = exp(log_prob_data - log_prob_data.max()).reshape(-1, 1)

                prob_prior = stats.burr12.pdf(sample_q, c=burr_c, d=burr_d, loc=0., scale=1.).reshape(-1, 1)
                prob_q = stats.burr12.pdf(sample_q, c=burr_c, d=burr_d, loc=0., scale=1.).reshape(-1, 1)

                prob_joint = prob_data * prob_prior
                w = prob_joint / prob_q
                w_bar = w / sum(w)

                std_valid = sqrt(sample_q[:, None, None]) * std_diag_valid[None, :, None]
                std_test = sqrt(sample_q[:, None, None]) * std_diag_test[None, :, None]

                nll_valid_burr12 = 0
                for i, valid_y_i in enumerate(y_valid):
                    nll_valid_burr12 += mc_nll(valid_y_i, mean_valid[:, i], std_valid[:, i], w_bar)
                nll_valid_burr12 /= valid_num

                nll_test_burr12 = 0
                for i, test_y_i in enumerate(y_test):
                    nll_test_burr12 += mc_nll(test_y_i, mean_test[:, i], std_test[:, i], w_bar)
                nll_test_burr12 /= test_num

                utils.print_yaml(type="burr12", **base_args,
                                 **{"burr_c": burr_c, "burr_d": burr_d,
                                    "valid-nll": round(float(nll_valid_burr12), 6),
                                    "test-nll": round(float(nll_test_burr12), 6)})
