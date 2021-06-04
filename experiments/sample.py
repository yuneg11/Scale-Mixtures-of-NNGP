import numpy as np
from scipy import stats

from jax import jit, random, vmap, grad
from jax import numpy as jnp
from jax.experimental import optimizers

from neural_tangents import stax
from neural_tangents.predict import gradient_descent_mse_ensemble

from tqdm import trange

from matplotlib import pyplot as plt
import seaborn as sns

from .ops import *


def add_subparser(subparsers):
    parser = subparsers.add_parser("sample", help="sample")
    parser.set_defaults(func=main)

    parser.add_argument("part", choices=["initial", "last", "full"])
    parser.add_argument("-trn", "--train-num", default=5, type=int)
    parser.add_argument("-tsn", "--test-num", default=40, type=int)
    parser.add_argument("-a", "--alpha", default=1., type=float)
    parser.add_argument("-b", "--beta", default=1., type=float)
    parser.add_argument("-s", "--seed", default=10, type=int)
    parser.add_argument("-ns", "--noise-scale", default=0.1, type=float)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-es", "--ensemble-size", default=1000, type=int)
    parser.add_argument("-ts", "--train-steps", default=10000, type=int)
    parser.add_argument("-fi", "--figure-index", default=20, type=int)
    parser.add_argument("-so", "--sample-output")
    parser.add_argument("-fo", "--figure-output")


def target_fn(x):
    return np.sin(x)


def get_kernel_fn_1():
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(1, W_std=1),
    )
    return jit(kernel_fn, static_argnums=(2,))


def get_kernel_fn_2():
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(1, W_std=1, b_std=0),
    )
    return jit(kernel_fn, static_argnums=(2,))


def partial(func, *args, **kwargs):
    return lambda key: func(key, *args, **kwargs)


def sample_network(key, alpha, beta, x_test):
    gamma_pure = random.gamma(key, a=alpha)
    gamma_rho = gamma_pure / beta
    invgamma = 1 / gamma_rho  # invgamma ~ invgamma(a = nu_q/2, scale = rho_q/2)
    sigma = sqrt(invgamma)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(1, W_std=sigma, b_std=0)
    )

    _, params = init_fn(key, (-1, 1)) 
    samples = apply_fn(params, x_test)

    return samples


def last_train_network(key, alpha, beta, x_train, y_train, x_test,
                       opt_init, opt_update, get_params, train_steps):
    gamma_pure = random.gamma(key, a=alpha)
    gamma_rho = gamma_pure / beta
    invgamma = 1 / gamma_rho  # invgamma ~ invgamma(a = nu_q/2, scale = rho_q/2)
    sigma = sqrt(invgamma)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(1, W_std=sigma, b_std=0)
    )

    _, params = init_fn(key, (-1, 1)) 
    opt_state = opt_init(params)

    loss = jit(lambda params, x, y: 0.5 * mean((apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    for i in trange(train_steps):
        grad_loss_whole = grad_loss(opt_state, x_train, y_train)

        grad_loss_front = [tuple([e * 0. for e in x]) for x in grad_loss_whole[:-1]]
        grad_loss_whole = grad_loss_front + [grad_loss_whole[-1]]

        opt_state = opt_update(i, grad_loss_whole, opt_state)

    params = get_params(opt_state)
    samples = apply_fn(params, x_test)

    return samples


def full_train_network(key, alpha, beta, x_train, y_train, x_test,
                       opt_init, opt_update, get_params, train_steps):
    gamma_pure = random.gamma(key, a=alpha)
    gamma_rho = gamma_pure / beta
    invgamma = 1 / gamma_rho  # invgamma ~ invgamma(a = nu_q/2, scale = rho_q/2)
    sigma = sqrt(invgamma)

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(512, W_std=8, b_std=0.05), stax.Erf(),
        stax.Dense(1, W_std=sigma, b_std=0)
    )

    _, params = init_fn(key, (-1, 1)) 
    opt_state = opt_init(params)

    loss = jit(lambda params, x, y: 0.5 * mean((apply_fn(params, x) - y) ** 2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    for i in trange(train_steps):
        grad_loss_whole = grad_loss(opt_state, x_train, y_train)
        opt_state = opt_update(i, grad_loss_whole, opt_state)

    params = get_params(opt_state)
    samples = apply_fn(params, x_test)

    return samples


def main(part, train_num, test_num, alpha, beta, seed, noise_scale, learning_rate,
         ensemble_size, train_steps, sample_output, figure_output, figure_index, **kwargs):

    if sample_output is None:
        if part == "initial":
            sample_output = "samples/{}-{:.1f}-{:.1f}.npy".format(part, alpha, beta)
        else:
            sample_output = "samples/{}-{:.1f}-{:.1f}-{}.npy".format(part, alpha, beta, train_steps)

    if figure_output is None:
        if part == "initial":
            figure_output = "images/{}-{:.1f}-{:.1f}-{}.png".format(part, alpha, beta, figure_index)
        else:
            figure_output = "images/{}-{:.1f}-{:.1f}-{}-{}.png".format(part, alpha, beta, figure_index, train_steps)

    key = random.PRNGKey(seed)
    key, x_key, y_key = random.split(key, 3)

    start, stop = (-np.pi, +np.pi)

    # Dataset
    x_train = random.uniform(x_key, minval=start, maxval=stop, shape=(train_num, 1))
    y_train = target_fn(x_train) + noise_scale * random.normal(y_key, shape=(train_num, 1))

    x_test = jnp.linspace(start, stop, num=test_num)[:, None]
    y_test = target_fn(x_test)

    x = jnp.concatenate([x_train, x_test], axis = 0)

    # Kernel
    if part == "initial":
        nu = 2 * alpha
        mu = zeros(test_num)
        kernel_fn_2 = get_kernel_fn_2()
        cov_2 = kernel_fn_2(x_test, x_test, get="nngp")
        cov = beta / alpha * cov_2
    elif part == "last":
        kernel_fn_1 = get_kernel_fn_1()
        predict_fn_1 = gradient_descent_mse_ensemble(kernel_fn_1, x_train, y_train, diag_reg=1e-4)
        mean_1, cov_1 = predict_fn_1(x_test=x_test, get="nngp", compute_cov=True)

        nu = 2 * alpha
        mu = mean_1.flatten()
        cov = beta / alpha * cov_1
    elif part == "full":
        kernel_fn_1 = get_kernel_fn_1()
        predict_fn_1 = gradient_descent_mse_ensemble(kernel_fn_1, x_train, y_train, diag_reg=1e-4)
        mean_1, cov_1 = predict_fn_1(x_test=x_test, get="ntk", compute_cov=True)

        nu = 2 * alpha
        mu = mean_1.flatten()
        cov = beta / alpha * cov_1

    std = sqrt(diag(cov))

    ensemble_key = random.split(key, ensemble_size)

    if part == "initial":
        network_fn = partial(sample_network, alpha, beta, x_test)
    elif part == "last":
        opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
        opt_update = jit(opt_update)

        network_fn = partial(
            last_train_network, alpha, beta, x_train, y_train, x_test,
            opt_init, opt_update, get_params, train_steps,
        )
    elif part == "full":
        opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
        opt_update = jit(opt_update)

        network_fn = partial(
            full_train_network, alpha, beta, x_train, y_train, x_test,
            opt_init, opt_update, get_params, train_steps,
        )

    samples = vmap(network_fn)(ensemble_key)
    samples = np.squeeze(samples, axis=2).T
    
    np.save(sample_output, (samples, nu, mu, std))
    print("Save samples to '{}'".format(sample_output))

    # Figure

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    u = np.linspace(-10, 10, 100)
    idx = figure_index
    student_t = stats.t.pdf(u, nu, loc=mu[idx], scale=std[idx])
    ax.set_xlim(-10, 10)
    sns.distplot(a=samples[idx], ax=ax, label="Sampled")
    ax.plot(u, student_t, label="Predicted", linewidth=3)
    ax.set_title("Correspondence")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right")

    fig.savefig(figure_output)
    print("Save figure to '{}'".format(figure_output))
