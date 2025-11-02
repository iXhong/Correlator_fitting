"""
Description: This module contains mathematical model functions used for data fitting.
@author GeorgeLiu
@since 2025
"""
import numpy as np

def one_cosh_func(params, t, T):
    A0 = params['A0']
    m0 = params['m0']
    return A0 * np.cosh(m0 * (t - T / 2))


def two_cosh_func(params, t, T):
    A0 = params['A0']
    A1 = params['A1']
    m0 = params['m0']
    m1 = params['m1']
    return A0 * np.cosh(m0 * (t - T / 2)) + A1 * np.cosh(m1 * (t - T / 2))


def one_exp(params, t):
    A0 = params['A0']
    m0 = params['m0']
    return A0 * np.exp(-m0 * t)
