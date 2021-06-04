import numpy as np
from scipy import stats

from jax import jit

from neural_tangents import stax
from neural_tangents.predict import gradient_descent_mse_ensemble

from . import data
from . import utils
from .ops import *


def add_subparser(subparsers):
    parser = subparsers.add_parser("classification", aliases=["cls"], help="classification")
    parser.set_defaults(func=main)

    parser.add_argument("dataset", choices=data.classification_datasets)
    parser.add_argument("test_dataset", nargs="?", choices=data.classification_datasets)
    parser.add_argument("-trn", "--train-num", default=3500, type=int)
    parser.add_argument("-tsn", "--test-num", default=1000, type=int)

    parser.add_argument("-nh", "--num-hiddens", default=1, type=int)
    parser.add_argument("-wv", "--w-variance", default=1., type=float)
    parser.add_argument("-bv", "--b-variance", default=0., type=float)
    parser.add_argument("-act", "--activation", default="relu", choices=["erf", "relu"])
    parser.add_argument("-e", "--epsilon-log-variance", default=5., type=float)
    parser.add_argument("-s", "--seed", default=10, type=int)

    parser.add_argument("-lv", "--last-layer-variance")
    
    parser.add_argument("-a", "--alpha")
    parser.add_argument("-b", "--beta")

    parser.add_argument("-bc", "--burr-c")
    parser.add_argument("-bd", "--burr-d")
    parser.add_argument("-sn", "--sample-num", default=1000, type=int)


def get_kernel_fn(depth, class_num, W_std, b_std, last_W_std=1., act="erf"):
    if act == "relu":
        act_class = stax.Relu
    elif act == "erf":
        act_class = stax.Erf
    else:
        raise KeyError("Unsupported act '{}'".format(act))

    layers = []
    for _ in range(depth):
        layers.append(stax.Conv(1, (3, 3), (1, 1), "SAME", W_std=W_std, b_std=b_std))
        layers.append(act_class())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(class_num, W_std=last_W_std))

    init_fn, apply_fn, kernel_fn = stax.serial(*layers)
    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    return kernel_fn


def gaussian_nll(y_test, mean, std):
    neg_log_prob = -stats.norm.logpdf(y_test, mean, std)
    return neg_log_prob


def student_t_nll(y_test, df, mean, std):
    neg_log_prob = -stats.t.logpdf(y_test, df, mean, std)
    return neg_log_prob


def mc_nll(y_test, mean, std, w_bar):
    prob = sum(array([
        stats.norm.logpdf(test_y_c, mean_c, std)
        for test_y_c, mean_c
        in zip(y_test, mean)
    ]), axis=0)
    neg_log_prob = -logsumexp(log(w_bar + 1e-24).flatten() + prob)
    return sum(neg_log_prob)


def main(dataset, test_dataset, train_num, test_num,
         num_hiddens, w_variance, b_variance, activation, burr_c, burr_d, alpha, beta,
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

    # Dataset

    x_train, y_train, x_test, y_test = data.get_dataset(
        dataset, train_num=train_num, test_num=test_num, normalize=True, seed=seed,
    )
    class_num = y_train.shape[1]

    if test_dataset is not None:
        _, _, x_test, y_test = data.get_dataset(
            test_dataset, train_num=1, test_num=test_num, normalize=True, seed=seed,
        )

    # Compute

    if invgamma or burr12:
        kernel_fn = get_kernel_fn(num_hiddens, class_num,
                                  W_std=W_std, b_std=b_std,
                                  last_W_std=1., act=activation)
        predict_fn = gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=eps)
        mean_test, cov_test = predict_fn(x_test=x_test, get="nngp", compute_cov=True)
        cov_train = kernel_fn(x_train, x_train, get="nngp")

        predict_label = argmax(mean_test, axis=1)
        true_label = argmax(y_test, axis=1)
        acc = mean(predict_label == true_label)

    if const:
        if (invgamma or burr12) and (-1e-10 <= last_W_std - 1. <= 1e-10):
            mean_test_const, cov_test_const = mean_test, cov_test
        else:
            kernel_fn_const = get_kernel_fn(num_hiddens, class_num,
                                            W_std=W_std, b_std=b_std,
                                            last_W_std=last_W_std, act=activation)
            predict_fn_const = gradient_descent_mse_ensemble(kernel_fn_const, x_train, y_train, diag_reg=eps)
            mean_test_const, cov_test_const = predict_fn_const(x_test=x_test, get="nngp", compute_cov=True)
        
        if not invgamma or not burr12:
            predict_label = argmax(mean_test_const, axis=1)
            true_label = argmax(y_test, axis=1)
            acc = mean(predict_label == true_label)

        std_diag_test_const = sqrt(diag(cov_test_const))

        nll_test_const = mean(array([
            gaussian_nll(y, nngp_mean, nngp_std)
            for y, nngp_mean, nngp_std
            in zip(y_test, mean_test_const, std_diag_test_const)
        ]))

        utils.print_yaml(type="const", **base_args,
                         **{"last-layer-var": last_layer_variance,
                            "test-nll": round(float(nll_test_const), 6),
                            "accuracy": round(float(acc), 6)})

    if invgamma:
        inv_cov_train = inv(cov_train + eps * eye(train_num))
        d_1_raw = sum(diag(matmul3(y_train.T, inv_cov_train, y_train)))

        for alpha in alpha_set:
            for beta in beta_set:
                nu = 2 * alpha
                cond_nu = nu + train_num * class_num
                d_1 = nu + alpha / beta * d_1_raw

                std_test = sqrt(diag(d_1 / cond_nu * beta / alpha * cov_test))

                nll_test_invgamma = mean(array([
                    student_t_nll(y, cond_nu, nngp_mean, std)
                    for y, nngp_mean, std
                    in zip(y_test, mean_test, std_test)
                ]))

                utils.print_yaml(type="invgamma", **base_args,
                                **{"alpha": alpha, "beta": beta,
                                   "test-nll": round(float(nll_test_invgamma), 6),
                                   "accuracy": round(float(acc), 6)})

    if burr12:
        minus_log_two_pi = -(train_num * class_num / 2) * log(2 * np.pi)
        minus_y_train_K_NNGP_y_train = -(1 / 2) * trace(matmul3(y_train.T, inv(cov_train + 1e-4 * eye(train_num)), y_train))
        
        pdf_cov = cov_train + 1e-4 * eye(train_num)
        nlogpdf = sum(array([
            stats.multivariate_normal.logpdf(y_train_c.reshape(train_num), None, pdf_cov, allow_singular=True)
            for y_train_c in y_train.T
        ]))
        minus_log_det_K_NNGP = nlogpdf - minus_log_two_pi - minus_y_train_K_NNGP_y_train

        std_diag_test = sqrt(diag(cov_test))

        for burr_c in burr_c_set:
            for burr_d in burr_d_set:
                sample_q = stats.burr12.rvs(c=burr_c, d=burr_d, loc=0., scale=1., size=sample_num, random_state=101)
                minus_log_sigma = -(1 / 2) * train_num * class_num * log(sample_q)

                log_prob_data = minus_log_two_pi + minus_log_det_K_NNGP + minus_y_train_K_NNGP_y_train / sample_q + minus_log_sigma
                prob_data = exp(log_prob_data - log_prob_data.max()).reshape(-1, 1)

                prob_prior = stats.burr12.pdf(sample_q, c=burr_c, d=burr_d, loc=0., scale=1.).reshape(-1, 1)
                prob_q = stats.burr12.pdf(sample_q, c=burr_c, d=burr_d, loc=0., scale=1.).reshape(-1, 1)

                prob_joint = prob_data * prob_prior
                w = prob_joint / prob_q
                w_bar = w / sum(w)

                std_test = sqrt(sample_q[:, None]) * std_diag_test[None, :]

                nll_test_burr12 = 0
                for i, test_y_i in enumerate(y_test):
                    nll_test_burr12 += mc_nll(test_y_i, mean_test[i], std_test[:, i], w_bar)
                nll_test_burr12 /= test_num * class_num

                utils.print_yaml(type="burr12", **base_args,
                                 **{"burr_c": burr_c, "burr_d": burr_d,
                                    "test-nll": round(float(nll_test_burr12), 6),
                                    "accuracy": round(float(acc), 6)})
