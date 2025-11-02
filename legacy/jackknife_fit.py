"""
Name:sample_fit_cosh
Author: George Liu
Since: 2025.9
Description: perform fitting & analysis on jk samples, return mean & err on paramters
"""


import numpy as np
import matplotlib.pyplot as plt
from latqcdtools.statistics.jackknife import jackknife
from lmfit import minimize,Parameters, fit_report
import glob


def data_load():    
    file_list = sorted(glob.glob("./mass/*.dat"))

    real_data_all = []

    for fname in file_list:
        data = np.loadtxt(fname, comments='#')
        filtered = data[data[:, 3] == 0]  # 第4列是 mumu
        real_values = filtered[:, 5]
        real_data_all.append(real_values)
    C_array = np.array(real_data_all)

    N_cfg = C_array.shape[0] # configuration num
    flip_data = (C_array[:,:48] + np.flip(C_array[:,-48:]))/2 #flip & average data
    t = np.arange(flip_data.shape[1])
    print(f'{N_cfg} 个组态，{len(t)} 个时间点')

    return flip_data, t, N_cfg


def fit_func_cosh(params,T,t):
    A0 = params['A0']
    m0 = params['m0']

    return A0*np.cosh(m0*(t-T/2))


def jk_mean_err(t,t_min,t_max,data):
    def func(data):
        return np.mean(data,axis=0)
    
    N_cfg = data.shape[0]

    mask = (t>=t_min) & (t<=t_max)
 
    jk_mean,jk_err = jackknife(f=func,data=data,numb_blocks=N_cfg,return_sample=False,conf_axis=0)

    t_fit = t[mask]
    data_fit = jk_mean[mask]
    yerr_fit = jk_err[mask]

    return data_fit,t_fit,yerr_fit


def get_fit_sample(t,t_min,t_max,data):
    def func(data):
        return np.mean(data,axis=0)
    
    N_cfg = data.shape[0]
    mask = (t>=t_min) & (t<=t_max)

    jk_samples,jk_mean,jk_err = jackknife(f=func,data=data,numb_blocks=N_cfg,return_sample=True,conf_axis=0)

    t_fit = t[mask]
    samp_fit = jk_samples[:,mask]
    yerr_fit = jk_err[mask]

    return t_fit,samp_fit,yerr_fit


def residual(params,data_fit,t_fit,yerr_fit,func,T):
    model = func(params,T,t_fit)
    return (data_fit - model)/yerr_fit


def one_state_fit(t_fit,data_fit,yerr_fit,T):
    params = Parameters()
    params.add('A0',value=data_fit[0],min=0)
    params.add('m0',value=0.5,min=0)

    out = minimize(residual,params=params,method='least_squares',kws={"data_fit":data_fit,"t_fit":t_fit,"yerr_fit":yerr_fit,"func":fit_func_cosh,"T":T})

    print(fit_report(out))

    return out


def sample_fit(t_fit,sample_fit,yerr_fit,T,N_cfg):
    params = Parameters()

    m0_list = []
    m0_err_list = []
    A0_list = []
    A0_err_list = []
    redchi2 = []
   
    for i in range(N_cfg):
        data_fit = sample_fit[i,:]
        params.add('A0',value=data_fit[0],min=0)
        params.add('m0',value=0.5,min=0)
        out = minimize(residual,params=params,method='least_square',kws={"data_fit":data_fit,"t_fit":t_fit,"yerr_fit":yerr_fit,"func":fit_func_cosh,"T":T})
        m0_list.append(out.params['m0'].value)
        m0_err_list.append(out.params['m0'].stderr)
        A0_list.append(out.params['A0'].value)
        A0_err_list.append(out.params['A0'].stderr)
        redchi2.append(out.redchi)

    # return m0_list,m0_err_list,A0_list,A0_err_list,redchi2
    return {
        'm0':m0_list,
        'm0_err':m0_err_list,
        'A0':A0_list,
        'A0_err':A0_err_list,
        'redchi2':redchi2
    }


def mean_func(data):
    return np.mean(data,axis=0)


def jackknife_estimate(sample_value):
    """
    return jackknife estimate value & the uncertainty 
    """
    sample_value = np.array(sample_value)
    N = len(sample_value)
    esti_value = np.mean(sample_value)
    esti_uncer = np.sqrt(N-1)*(np.mean(sample_value**2)-(np.mean(sample_value))**2)
    return esti_value,esti_uncer


def main(tmin,tmax,T,show_result=False):

    data,t,N_cfg = data_load()
    data_mean = np.mean(data,axis=0)

    t_fit,samp_fit,yerr_fit = get_fit_sample(t,t_min=tmin,t_max=tmax,data=data)
    result = sample_fit(t_fit=t_fit,sample_fit=samp_fit,yerr_fit=yerr_fit,T=T,N_cfg=N_cfg)

    m0_list = result['m0']
    m0_err = result['m0_err']
    A0_list = result['A0']
    A0_err = result['A0_err']

    #计算jackknife fit的m0以及uncertainty
    m0, m0_uncer = jackknife_estimate(m0_list)
    A0, A0_uncer = jackknife_estimate(A0_list)
    if show_result:
        print("m0:")
        print(m0)
        print(m0_uncer)
        print("A0:")
        print(A0)
        print(A0_uncer)

    return {
        "m0":m0,
        "m0_err":m0_err
    }


if __name__ == "__main__":
    
    out = main(tmin=5,tmax=16,T=96)
    print(out['m0'])
    



