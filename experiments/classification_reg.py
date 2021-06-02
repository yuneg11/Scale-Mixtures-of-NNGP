from jax import numpy as jnp
from scipy import stats

from neural_tangents.predict import gradient_descent_mse_ensemble

from . import data
from .ops import *
from .classification_utils import *


def add_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("classification-reg",
                                   aliases=["cls_reg"],
                                   help="classification-reg",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=main)

    parser.add_argument("-d", "--dataset", choices=data.classification_datasets)
    parser.add_argument("-td", "--test-dataset", choices=data.classification_datasets)
    parser.add_argument("-trn", "--train-num", default=3500, type=int)
    parser.add_argument("-tsn", "--test-num", default=1000, type=int)
    parser.add_argument("-nh", "--num-hiddens", default=1, type=int)
    parser.add_argument("-wv", "--w-variance", default=1., type=float)
    parser.add_argument("-bv", "--b-variance", default=0., type=float)
    parser.add_argument("-act", "--activation", default="relu", choices=["erf", "relu"])
    parser.add_argument("-a", "--alpha", default=2., type=float)
    parser.add_argument("-b", "--beta", default=2., type=float)
    parser.add_argument("-e", "--epsilon-log-variance", default=8., type=float)
    parser.add_argument("-lv", "--last-layer-variance", default=1., type=float)
    parser.add_argument("-s", "--seed", default=10, type=int)


def gaussian_nll(y_test, mean, std):
    neg_log_prob = -stats.norm.logpdf(y_test, mean, std)
    return neg_log_prob


def studentt_nll(y_test, freedom, mean, std):
    neg_log_prob = -stats.t.logpdf(y_test, freedom, mean, std)
    return neg_log_prob


def main(dataset, test_dataset, num_hiddens, w_variance, b_variance, activation,
         alpha, beta, epsilon_log_variance, last_layer_variance, seed, train_num,
         test_num, **kwargs):

    raw_epsilon_log_variance = epsilon_log_variance
    epsilon_log_variance = -6 + epsilon_log_variance / 2
    epsilon_variance = jnp.power(10, epsilon_log_variance)
    # last_layer_variance = 2 * beta / (2 * alpha - 2)

    normalize = True

    # dataset
    x_train, y_train, x_test, y_test = data.get_dataset(
        dataset,
        train_num=train_num,
        test_num=test_num,
        normalize=normalize,
        seed=seed
    )
    class_num = y_train.shape[1]

    if test_dataset is not None:
        _, _, x_test, y_test = data.get_dataset(
            test_dataset,
            train_num=1,
            test_num=test_num,
            normalize=normalize,
            seed=seed
        )

    # Models
    const_kernel_fn = get_cnn_kernel(num_hiddens, class_num, activation,
                                   W_std=sqrt(w_variance),
                                   b_std=sqrt(b_variance),
                                   last_W_std=sqrt(last_layer_variance))

    inv_kernel_fn = get_cnn_kernel(num_hiddens, class_num, activation,
                                     W_std=sqrt(w_variance),
                                     b_std=sqrt(b_variance),
                                     last_W_std=1.)

    # TODO: Only when last layer variance is 1.
    assert -1e-6 <= last_layer_variance - 1. <= 1e-6
    const_predict_fn = gradient_descent_mse_ensemble(const_kernel_fn, x_train, y_train, diag_reg=epsilon_variance)
    const_nngp_mean_test, const_nngp_covariance_test = const_predict_fn(x_test=x_test, get="nngp", compute_cov=True)
    const_nngp_std_test = sqrt(diag(const_nngp_covariance_test))

    inv_nngp_covariance_test = const_nngp_covariance_test

    # TODO: last layer variance != 1
    # const_predict_fn = gradient_descent_mse_ensemble(const_kernel_fn, x_train, y_train, diag_reg=epsilon_variance)
    # const_nngp_mean_test, const_nngp_covariance_test = const_predict_fn(x_test=x_test, get="nngp", compute_cov=True)
    # const_nngp_std_test = sqrt(diag(const_nngp_covariance_test))

    # inv_predict_fn = gradient_descent_mse_ensemble(inv_kernel_fn, x_train, y_train, diag_reg=epsilon_variance)
    # _, inv_nngp_covariance_test = inv_predict_fn(x_test=x_test, get="nngp", compute_cov=True)

    predict_label = argmax(const_nngp_mean_test, axis=1)
    true_label = argmax(y_test, axis=1)

    correct_count = sum(predict_label == true_label)
    acc = correct_count / test_num

    test_neg_log_prob_constant = mean(array([
        gaussian_nll(y, nngp_mean, nngp_std)
        for y, nngp_mean, nngp_std
        in zip(y_test, const_nngp_mean_test, const_nngp_std_test)
    ]))

    const_kernel = const_kernel_fn(x_train, x_train, "nngp")
    inverse_k_11_const = inv(const_kernel + epsilon_variance * eye(train_num))
    d_1_const = sum(diag(matmul3(y_train.T, inverse_k_11_const, y_train)))

    # Test
    for alpha in [0.5, 1., 2., 4.]:
        for beta in [0.5, 1., 2., 4.]:
            nu = 2 * alpha
            conditional_nu = nu + train_num * class_num
            d_1 = nu + d_1_const * alpha / beta
            conditional_kernel = d_1 / conditional_nu * beta / alpha * inv_nngp_covariance_test

            test_std = sqrt(diag(conditional_kernel))

            test_neg_log_prob_invgamma = mean(array([
                studentt_nll(y, conditional_nu, nngp_mean, std)
                for y, nngp_mean, std
                in zip(y_test, const_nngp_mean_test, test_std)
            ]))

            print("------------------------------------------------------------------")
            print("num_hiddens: {:<2d}  / act:   {}".format(num_hiddens, activation))
            print("w_variance:  {:1.1f} / alpha: {:1.1f}".format(w_variance, alpha))
            print("b_variance:  {:1.1f} / beta:  {:1.1f}".format(b_variance, beta))
            print("epsilon_log_variance: {}".format(raw_epsilon_log_variance))
            print("last_layer_variance: {}     / seed: {}".format(last_layer_variance, seed))
            print("---------------------------------------------")
            print("Test NLL for invgamma prior:  [{:13.8f}]".format(test_neg_log_prob_invgamma))
            print("Test NLL for constant prior:  [{:13.8f}]".format(test_neg_log_prob_constant))
            print("Accuracy: {:.4f}".format(acc))
