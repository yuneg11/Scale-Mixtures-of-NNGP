import numpy

import jax.numpy as np

from jax import random
from jax.experimental import optimizers
from jax.api import jit, grad

from neural_tangents import stax
from neural_tangents.predict import gradient_descent_mse_ensemble

from datetime import datetime
import os

from examples import datasets
import warnings

from utils import *

warnings.filterwarnings("ignore")
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.20'
now = datetime.now()


# def mean_covariance(batch, induced, kernel_fn, inducing_mu, inducing_sigma):
#     batch_len = len(batch)
#     induced_len = len(induced)

#     batch_induced = np.concatenate([batch, induced], axis=0)
#     predict_fn = gradient_descent_mse_ensemble(kernel_fn, induced, np.zeros((induced_len, class_num)), diag_reg=1e-4)
#     _, B_B = predict_fn(x_test=batch, get='nngp', compute_cov=True)

#     kernel = kernel_fn(batch_induced, batch_induced, 'nngp')
#     k_b_b, k_b_i, k_i_b, k_i_i = split_kernel(kernel, batch_len)
#     k_i_i_inverse = np.linalg.inv(k_i_i + 1e-4 * np.eye(induced_len))

#     A_B = np.matmul(k_b_i, k_i_i_inverse)
#     L = np.linalg.cholesky(k_i_i)
#     A_B_L = np.matmul(A_B, L)
#     A_B_L = kron_diag(A_B_L, n=class_num)
#     B_B = kron_diag(B_B, n=class_num)

#     mean = np.matmul(A_B_L, inducing_mu)
#     covariance = matmul(A_B_L, inducing_sigma, A_B_L.T) + B_B
#     covariance_L = np.linalg.cholesky(covariance)

#     return mean, covariance_L


def sampling_f_b(sample_num, batch_num):
    mean = np.zeros(batch_num * class_num)
    kernel = np.eye(batch_num * class_num)
    sample = random.multivariate_normal(key_batch, mean, kernel, shape=(sample_num,))
    return sample


def KL(inducing_mu, inducing_sigma, inducing_num):
    kl = 1/2 * (-np.log(np.linalg.det(inducing_sigma)) - inducing_num * class_num + np.trace(inducing_sigma) + np.matmul(inducing_mu, inducing_mu))
    return kl


def get_true_label_values(label, values, sample_num):
    label_argmax = np.argmax(label, axis=1)
    label_idxs = np.tile(label_argmax, sample_num)[:, np.newaxis]
    true_label_values = np.take_along_axis(values.reshape(-1, class_num), label_idxs, axis=1).reshape(sample_num, -1)
    return true_label_values


def loglikelihood(label, sampled_f, sample_num, batch_num, training_num):
    sampled_f = np.transpose(sampled_f.reshape(sample_num, -1, batch_num), axes=(0, 2, 1))
    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_label_values(label, sampled_f_softmax, sample_num)
    log_likelihood = training_num * np.mean(np.mean(true_label_softmax, axis=1))
    return log_likelihood


def loglikelihood3(label, sampled_f, sample_num, batch_num):
    sampled_f = np.transpose(sampled_f.reshape(sample_num, -1, batch_num), axes=(0, 2, 1))
    sampled_f_softmax = log_softmax(sampled_f)
    true_label_softmax = get_true_label_values(label, sampled_f_softmax, sample_num)
    log_likelihood = np.sum(logsumexp(true_label_softmax.T) - np.log(sample_num))
    return log_likelihood


def ELBO(inducing_points, inducing_mu, inducing_sigma, sample_num, inducing_num, batch_num, training_num):
    inducing_sigma = np.diag(inducing_sigma)
    index = numpy.random.choice(training_num, batch_num, replace=False)
    batch_x_train = x_train[index]
    batch_y_train = y_train[index]

    mean, covariance_L = mean_covariance(batch_x_train, inducing_points, kernel_fn_1, inducing_mu, inducing_sigma)
    sample = sampling_f_b(sample_num, batch_num,)
    sample = mean.reshape(-1,1) + np.matmul(covariance_L, sample.T)
    sample = sample.T
    log_likelihood = loglikelihood(batch_y_train, sample, sample_num, batch_num, training_num)
    kl = KL(inducing_mu, inducing_sigma, inducing_num)
    elbo = log_likelihood - kl
    return -elbo


# define hyperparameters
training_sample = 60000
test_sample = 10000
training_num = 1000
test_num = 10
class_num = 10

inducing_num = 100

learning_rate = 0.000008
training_steps = 15000

sample_num = 100
batch_num = 32

test_f_sample_num = 10000


# download mnist datasets
x_train, y_train, x_test, y_test = datasets.get_dataset('mnist', training_sample, test_sample,
                                                        do_flatten_and_normalize=False)

x_train = x_train[:training_num]
y_train = y_train[:training_num]
x_test = x_test[:test_num]
y_test = y_test[:test_num]

# normalize mnist datasets
x_train = normalize(x_train)
x_test = normalize(x_test)

# define layers before last layer
layers = []
for _ in range(3):
    layers.append(stax.Conv(1, (3, 3), (1, 1), 'SAME'))
    layers.append(stax.Relu())
layers.append(stax.Flatten())
layers.append(stax.Dense(class_num, W_std=1))

init_fn_1, apply_fn_1, kernel_fn_1 = stax.serial(*layers)
apply_fn_1 = jit(apply_fn_1)
kernel_fn_1 = jit(kernel_fn_1, static_argnums=(2,))
predict_fn_1 = gradient_descent_mse_ensemble(kernel_fn_1, x_train, y_train, diag_reg=1e-4)


key = random.PRNGKey(0)
key, key_induce, key_batch, key_test = random.split(key, 4)

inducing_points = x_train[:inducing_num]
# inducing_points = random.normal(key_induce,(inducing_num,28,28,1))
inducing_mu = np.zeros(inducing_num * class_num)
inducing_sigma = np.ones(inducing_num * class_num)

ELBO = jit(ELBO, static_argnums=(3, 5, 6))
grad_elbo = grad(ELBO, argnums=(0, 1, 2))
grad_elbo = jit(grad_elbo, static_argnums=(3, 5, 6))

opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_update = jit(opt_update)
opt_state = opt_init((inducing_points, inducing_mu, inducing_sigma))

print(ELBO(inducing_points, inducing_mu, inducing_sigma, sample_num, inducing_num, batch_num, training_num))

for i in range(training_steps):
    if (i + 1) % 100 == 0:
        inducing_points, inducing_mu, inducing_sigma = get_params(opt_state)
        elbo_now = ELBO(inducing_points, inducing_mu, inducing_sigma, sample_num, inducing_num, batch_num, training_num) / training_num
        print('{}/{}th training_step, elbo : {}'.format(i + 1, training_steps, elbo_now))
    opt_state = opt_update(i, grad_elbo(*get_params(opt_state), sample_num, inducing_num, batch_num, training_num), opt_state)
inducing_points, inducing_mu, inducing_sigma = get_params(opt_state)

print(ELBO(inducing_points, inducing_mu, inducing_sigma, sample_num, inducing_num, batch_num, training_num))

induced_test = np.concatenate([inducing_points, x_test], axis=0)
test_kernel = kernel_fn_1(induced_test, induced_test, 'nngp')

kernel_i_i, kernel_i_t, kernel_t_i, kernel_t_t = split_kernel(test_kernel, inducing_num)
kernel_i_i_inverse = np.linalg.inv(kernel_i_i + 1e-4 * np.eye(inducing_num))

L_induced = np.linalg.cholesky(kernel_i_i)

A = matmul(kernel_t_i, kernel_i_i_inverse, L_induced)
A = kron_diag(A, n=class_num)

L_induced = kron_diag(L_induced, n=class_num)
L_mu = np.matmul(L_induced, inducing_mu)

predict_fn_1 = gradient_descent_mse_ensemble(kernel_fn_1, inducing_points, np.transpose(L_mu.reshape(-1, inducing_num)), diag_reg=1e-4)
test_mean, test_covariance = predict_fn_1(x_test=x_test, get='nngp', compute_cov=True)
test_mean = test_mean.T.reshape(-1)
test_covariance = kron_diag(test_covariance, n=class_num)

inducing_sigma = np.diag(inducing_sigma)
A_sigma = matmul(A, inducing_sigma, A.T)
test_sigma = A_sigma + test_covariance

test_f_sample = random.multivariate_normal(key_test, test_mean, test_sigma, shape=(test_f_sample_num,))

test_loglikelihood = -loglikelihood3(y_test, test_f_sample, test_f_sample_num, test_num) / test_num

print(test_loglikelihood)
