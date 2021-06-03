import numpy as np

from scipy import stats

from jax import jit
from jax import numpy as jnp
from jax.numpy import linalg

from neural_tangents import stax
from neural_tangents.predict import gradient_descent_mse_ensemble

from . import data
from .ops import *
from .classification_utils import *


def add_subparser(subparsers):
    from argparse import ArgumentDefaultsHelpFormatter as DefaultsFormatter

    parser = subparsers.add_parser("importance-sampling-classification",
                                   aliases=["isc"],
                                   help="importance-sampling-classification",
                                   formatter_class=DefaultsFormatter)
    parser.set_defaults(func=main)

    parser.add_argument("-d", "--dataset", choices=data.classification_datasets)
    parser.add_argument("-td", "--test-dataset", choices=data.classification_datasets)
    parser.add_argument("-trn", "--train-num", default=3500, type=int)
    parser.add_argument("-tsn", "--test-num", default=1000, type=int)
    parser.add_argument("-ds", "--dist", default="burr12")
    parser.add_argument("-sn", "--sample-num", default=1000, type=int)
    parser.add_argument("-nh", "--num-hiddens", default=1, type=int)
    parser.add_argument("-wv", "--w-variance", default=1., type=float)
    parser.add_argument("-bv", "--b-variance", default=0., type=float)
    parser.add_argument("-act", "--activation", default="relu", choices=["erf", "relu"])
    parser.add_argument("-e", "--epsilon-log-variance", default=8., type=float)
    parser.add_argument("-s", "--seed", default=10, type=int)
    parser.add_argument("-bc", "--burr-c", default=2., type=float)
    parser.add_argument("-bd", "--burr-d", default=1., type=float)


def main(dataset, num_hiddens, w_variance, b_variance, activation, burr_c, burr_d,
         epsilon_log_variance, seed, dist, sample_num,
         train_num, test_num, test_dataset, last_layer_variance=None, **kwargs):

    raw_epsilon_log_variance = epsilon_log_variance
    epsilon_variance = power(10, -6 + epsilon_log_variance / 2)

    normalize = True

    # dataset
    train_x, train_y, test_x, test_y = data.get_dataset(
        dataset,
        train_num=train_num,
        test_num=test_num,
        normalize=normalize,
        seed=seed
    )
    class_num = train_y.shape[1]

    if test_dataset is not None:
        _, _, test_x, test_y = data.get_dataset(
            test_dataset,
            train_num=1,
            test_num=test_num,
            normalize=normalize,
            seed=seed
        )

    # Models
    W_std = sqrt(w_variance)
    b_std = sqrt(b_variance)

    kernel_fn_is = get_cnn_kernel(num_hiddens, class_num, activation, W_std=W_std, b_std=b_std, last_W_std=1.)

    # Importance

    nngp_train_covariance = kernel_fn_is(train_x, train_x, 'nngp')  # K_NNGP(X_D, X_D)

    predict_fn = gradient_descent_mse_ensemble(kernel_fn_is, train_x, train_y, diag_reg=epsilon_variance)
    nngp_mean_is_test, nngp_covariance_is_test = predict_fn(x_test=test_x, get="nngp", compute_cov=True)

    nngp_std_diag_is_test = jnp.sqrt(jnp.diag(nngp_covariance_is_test))

    minus_log_two_pi = -(train_num * class_num / 2) * log(2 * jnp.pi)
    minus_y_train_K_NNGP_y_train = -(1 / 2) * trace(matmul3(train_y.T, inv(nngp_train_covariance + 1e-4 * eye(train_num)), train_y))
    nlogpdf = sum(array([
        stats.multivariate_normal.logpdf(train_y_c.reshape(train_num), None, nngp_train_covariance + 1e-4 * eye(train_num), allow_singular=True)
        for train_y_c in train_y.T
    ]))

    minus_log_det_K_NNGP = nlogpdf - minus_log_two_pi - minus_y_train_K_NNGP_y_train

    for burr_c in [0.5, 1., 2., 3., 4.]:
        for burr_d in [0.5, 1., 2., 3., 4.]:
            sample_q = stats.burr12.rvs(c=burr_c, d=burr_d, loc=0., scale=1., size=(sample_num), random_state=101)
            minus_log_sigma = -(1 / 2) * train_num * class_num * log(sample_q)

            log_probability_data = minus_log_two_pi + minus_log_det_K_NNGP + minus_y_train_K_NNGP_y_train / sample_q + minus_log_sigma
            log_probability_data = log_probability_data - log_probability_data.max()
            probability_data = exp(log_probability_data).reshape(-1, 1)

            probability_prior = stats.burr12.pdf(sample_q, c=burr_c, d=burr_d, loc=0., scale=1.).reshape(-1, 1)#[:, None]
            probability_q  = stats.burr12.pdf(sample_q, c=burr_c, d=burr_d, loc=0., scale=1.).reshape(-1, 1)#[:, None]

            probability_joint = probability_data * probability_prior
            w = probability_joint / probability_q
            w_bar = w / sum(w)
            
            nngp_std_is_test = sqrt(sample_q[:, None]) * nngp_std_diag_is_test[None, :]

            def mcnll_is(test_y, mean, std, w_bar):
                prob = sum(array([
                    stats.norm.logpdf(test_y_c, mean_c, std)
                    for test_y_c, mean_c
                    in zip(test_y, mean)
                ]), axis=0)
                neg_log_prob = -logsumexp(log(w_bar + 1e-24).flatten() + prob)
                return sum(neg_log_prob)

            # calculate nll for test points
            neg_log_prob_is_test = 0
            for i, test_y_i in enumerate(test_y):
                neg_log_prob_is_test += mcnll_is(test_y_i, nngp_mean_is_test[i], nngp_std_is_test[:, i], w_bar)
            neg_log_prob_is_test /= test_num * class_num

            # test_neg_log_prob_constant = mean(array([
            #     mcnll_is(y, nngp_mean, nngp_std, w_bar)
            #     for y, nngp_mean, nngp_std
            #     in zip(test_y, nngp_mean_is_test, nngp_std_is_test.T)
            # ]))

            predict_label = argmax(nngp_mean_is_test, axis=1)
            true_label = argmax(test_y, axis=1)

            correct_count = sum(predict_label == true_label)
            acc = correct_count / test_num


            print("------------------------------------------------------------------")
            print("num_hiddens: {:<2d}  / act:    {}".format(num_hiddens, activation))
            print("w_variance:  {:1.1f} / burr-c: {:1.1f}".format(w_variance, burr_c))
            print("b_variance:  {:1.1f} / burr-d: {:1.1f}".format(b_variance, burr_d))
            print("epsilon_log_variance: {}".format(raw_epsilon_log_variance))
            print("last_layer_variance: {}     / seed: {}".format(last_layer_variance, seed))
            print("---------------------------------------------")
            print("Test NLL for {} prior:   [{:13.8f}]".format(dist, neg_log_prob_is_test))
            print("Accuracy: {:.4f}".format(acc))
