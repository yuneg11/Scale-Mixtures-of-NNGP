from collections.abc import Iterable

from .ops import sqrt


def process_invgamma_args(alpha, beta):
    if alpha is not None and beta is not None:
        invgamma = True

        try:
            alpha_set = eval(alpha)
            if not isinstance(alpha_set, Iterable):
                alpha_set = [alpha_set]
        except:
            raise ValueError("Unable to evaluate alpha values '{}'".format(alpha))
        
        try:
            beta_set = eval(beta)
            if not isinstance(beta_set, Iterable):
                beta_set = [beta_set]
        except:
            raise ValueError("Unable to evaluate beta values '{}'".format(beta))
    else:
        invgamma = False
        alpha_set = None
        beta_set = None
    
    return invgamma, alpha_set, beta_set


def process_burr12_args(burr_c, burr_d):
    if burr_c is not None and burr_d is not None:
        burr12 = True

        try:
            burr_c_set = eval(burr_c)
            if not isinstance(burr_c_set, Iterable):
                burr_c_set = [burr_c_set]
        except:
            raise ValueError("Unable to evaluate burr_c values '{}'".format(burr_c))
        
        try:
            burr_d_set = eval(burr_d)
            if not isinstance(burr_d_set, Iterable):
                burr_d_set = [burr_d_set]
        except:
            raise ValueError("Unable to evaluate burr_d values '{}'".format(burr_d))
    else:
        burr12 = False
        burr_c_set = None
        burr_d_set = None

    return burr12, burr_c_set, burr_d_set


def process_const_args(last_layer_variance):
    if last_layer_variance is not None:
        const = True
        last_W_std = sqrt(float(last_layer_variance))
    else:
        const = False
        last_W_std = None

    return const, last_W_std


def print_yaml(**kwargs):
    first = True
    
    for key, value in kwargs.items():
        if first:
            print("- {}: {}".format(key, value))
        else:
            print("  {}: {}".format(key, value))
        
        first = False
