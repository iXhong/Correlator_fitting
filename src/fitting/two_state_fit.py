"""
two state fit
@author: George Liu
@since 2025.10
"""
from lmfit import minimize, Parameters


def one_state_fit(t_fit, data_fit, err_fit,T,residual,scale_factor=1):
    params = Parameters()
    params.add("A0", value=data_fit[0]*scale_factor, min=0)
    params.add("m0", value=0.5, min=0)

    result = minimize(residual, params, method='least_squares',
                      kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit, "T": T})
        
    return result


def two_state_fit(t_fit, data_fit, err_fit, T,residual,scale_factor=1):
    params = Parameters()
    params.add('A0', value=1e-15 * scale_factor, min=1e-16 * scale_factor)
    params.add('A1', value=1e-17 * scale_factor, min=0, max=1e-7 * scale_factor)
    params.add('m0', value=0.60, min=0.6, max=0.8)
    params.add('m1', value=1.0, min=0.7, max=5.0)

    result = minimize(residual, params, method='least_squares',
                      kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit, "T": T})

    return result


