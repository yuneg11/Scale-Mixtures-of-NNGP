from scipy import stats

from jax import jit, grad, random
from jax.scipy.special import gammaln, digamma

from neural_tangents.predict import gradient_descent_mse_ensemble

from tqdm import tqdm

from .ops import *
from .utils import *
from .classification_utils import *


def softplus(x):
    return log(1 + exp(x))


def sample_f_b(invgamma_a, invgamma_b, sample_num, batch_num, class_num, key):
    _, key_gamma, key_normal = random.split(key, 3)

    mean = zeros(batch_num * class_num)
    covariance = eye(batch_num * class_num)
    sampled_f = random.multivariate_normal(key_normal, mean, covariance, shape=(sample_num,))

    gamma_pure = random.gamma(key_gamma, a=softplus(invgamma_a))
    gamma_rho = gamma_pure / softplus(invgamma_b)
    invgamma = 1 / gamma_rho  # invgamma ~ invgamma(a = nu_q/2, scale = rho_q/2)
    sigma = sqrt(invgamma)
    sampled_f = sampled_f * sigma

    return sampled_f


def kl_divergence(
    inducing_mu, inducing_sigma_mat,
    invgamma_a, invgamma_b, alpha, beta,
    class_num, induce_num, inducing_points, kernel_fn, kernel_scale
):

    k_i_i = kernel_fn(inducing_points, inducing_points, "nngp") * kernel_scale
    k_i_i_inverse = inv(k_i_i + 1e-6 * eye(induce_num))
    k_i_i_inverse = kron_diag(k_i_i_inverse, n=class_num)

    inducing_mu_t = inducing_mu[None, ...]
    inducing_mu_vec = inducing_mu[..., None]

    kl = 1 / 2 * (softplus(invgamma_a) / softplus(invgamma_b) * matmul3(inducing_mu_t, k_i_i_inverse, inducing_mu_vec) \
                  + trace(matmul(k_i_i_inverse, inducing_sigma_mat)) \
                  + (logdet(k_i_i) - logdet(inducing_sigma_mat)) \
                  - (induce_num * class_num)) \
       + alpha * log(softplus(invgamma_b) / beta) \
       - gammaln(softplus(invgamma_a)) + gammaln(alpha) \
       + (softplus(invgamma_a) - alpha) * digamma(softplus(invgamma_a)) \
       + (beta - softplus(invgamma_b)) * softplus(invgamma_a) / softplus(invgamma_b)

    return sum(kl)  # convert to scalar


def negative_elbo(
    x_batch, y_batch,
    kernel_fn, kernel_scale,
    inducing_mu, inducing_sigma, inducing_points,
    invgamma_a, invgamma_b, alpha, beta,
    train_num, class_num, sample_num, induce_num, batch_num,
    key,
):
    inducing_sigma_mat = diag(softplus(inducing_sigma))

    mean, covariance_L = mean_covariance(x_batch, inducing_points, kernel_fn,
                                         inducing_mu, inducing_sigma_mat,
                                         batch_num, induce_num, class_num, kernel_scale)

    sampled_f = sample_f_b(invgamma_a, invgamma_b, sample_num, batch_num, class_num, key)
    sampled_f = (mean.reshape(-1, 1) + matmul(covariance_L, sampled_f.T)).T
    sampled_f = transpose(sampled_f.reshape(sample_num, class_num, batch_num), axes=(0, 2, 1))

    ll = log_likelihood(y_batch, sampled_f, train_num)
    kl = kl_divergence(inducing_mu, inducing_sigma_mat,
                       invgamma_a, invgamma_b, alpha, beta,
                       class_num, induce_num, inducing_points, kernel_fn, kernel_scale)

    n_elbo = (- ll + kl) / train_num
    return n_elbo


def test_nll_acc(
    x_test, y_test,
    kernel_fn, kernel_scale,
    inducing_mu, inducing_sigma, inducing_points,
    invgamma_a, invgamma_b,
    test_num, class_num, test_sample_num, induce_num,
    key,
):
    inducing_sigma_mat = diag(softplus(inducing_sigma))

    test_nll_list = []
    total_correct_count = 0

    test_batches = TestBatch(x_test, y_test, 64)
    for batch_x, batch_y in tqdm(test_batches, leave=False, ncols=0):
        batch_num = batch_x.shape[0]

        induced_test = concatenate([inducing_points, batch_x], axis=0)
        kernel = kernel_fn(induced_test, induced_test, "nngp") * kernel_scale

        kernel_i_i, kernel_i_t, kernel_t_i, kernel_t_t = split_kernel(kernel, induce_num)
        kernel_i_i_inverse = inv(kernel_i_i + 1e-6 * eye(induce_num))

        # L_induced = jnp.linalg.cholesky(kernel_i_i)
        # L_induced = kron_diag(L_induced, n=class_num)
        # L_mu = matmul(L_induced, inducing_mu)
        L_mu = inducing_mu

        # A_L = matmul3(kernel_t_i, kernel_i_i_inverse, L_induced)
        A_L = matmul(kernel_t_i, kernel_i_i_inverse)
        A_L = kron_diag(A_L, n=class_num)

        inducing_x = inducing_points
        inducing_y = transpose(L_mu.reshape(-1, induce_num))
        predict_fn = gradient_descent_mse_ensemble(kernel_fn, inducing_x, inducing_y, diag_reg=1e-6)
        test_mean, test_covariance = predict_fn(x_test=batch_x, get="nngp", compute_cov=True)
        test_covariance *= kernel_scale

        test_mean = test_mean.T.flatten()
        test_covariance = kron_diag(test_covariance, n=class_num)

        A_sigma = matmul3(A_L, inducing_sigma_mat, A_L.T)
        test_sigma = A_sigma + test_covariance

        key, key_test = random.split(key)

        test_f_sample = stats.multivariate_t.rvs(
            random_state=key_test[1],
            loc=test_mean,
            shape=test_sigma,
            df=2 * softplus(invgamma_a),
            size=(test_sample_num,),
        ).reshape(test_sample_num, class_num, batch_num)
        test_f_sample = transpose(test_f_sample, axes=(0, 2, 1))

        batch_test_nll = -test_log_likelihood(batch_y, test_f_sample, test_sample_num)
        test_nll_list.append(batch_test_nll)

        batch_correct_count = get_correct_count(batch_y, test_f_sample)
        total_correct_count += batch_correct_count

    nll = sum(array(test_nll_list)) / test_num
    acc = total_correct_count / test_num

    return nll, acc


def get_train_vars(inducing_mu, inducing_sigma, inducing_points, invgamma_a, invgamma_b):
    train_params = (inducing_mu, inducing_sigma, inducing_points, invgamma_a, invgamma_b)

    negative_elbo_jit = jit(negative_elbo, static_argnums=(2, 3, 9, 10, 11, 12, 13, 14, 15))
    grad_elbo = grad(negative_elbo_jit, argnums=(4, 5, 6, 7, 8))
    grad_elbo_jit = jit(grad_elbo, static_argnums=(2, 3, 9, 10, 11, 12, 13, 14, 15))

    return train_params, negative_elbo_jit, grad_elbo_jit
