"""
Name:sample_fit_cosh
Author: George Liu
Since: 2025.9
Description: perform fitting & analysis on jk samples, return mean & err on paramters
"""


import numpy as np
import matplotlib.pyplot as plt
from latqcdtools.statistics.jackknife import jackknife
from latqcdtools.statistics.bootstr import bootstr
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


def jackknife_fit(t,t_min,t_max,data):
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
    A0_list = []
   
    for i in range(N_cfg):
        data_fit = sample_fit[i,:]
        params.add('A0',value=data_fit[0],min=0)
        params.add('m0',value=0.5,min=0)
        out = minimize(residual,params=params,method='least_square',kws={"data_fit":data_fit,"t_fit":t_fit,"yerr_fit":yerr_fit,"func":fit_func_cosh,"T":T})
        m0_list.append(out.params['m0'].value)
        A0_list.append(out.params['A0'].value)

    return m0_list,A0_list


def mean_func(data):
    return np.mean(data,axis=0)


if __name__ == "__main__":

    tmin,tmax = 5,16
    T=96

    data,t,N_cfg = data_load()

    t_fit,samp_fit,yerr_fit = get_fit_sample(t,t_min=tmin,t_max=tmax,data=data)

    m0_list,A0_list = sample_fit(t_fit=t_fit,sample_fit=samp_fit,yerr_fit=yerr_fit,T=T,N_cfg=N_cfg)

    m0_list = np.array(m0_list)
    A0_list = np.array(A0_list)

    m0_mean,m0_err = bootstr(mean_func,data=m0_list,numb_samples=1000)
    A0_mean,A0_err = bootstr(mean_func,data=A0_list,numb_samples=1000)

    print("="*40)
    print(f"m0_mean = {m0_mean:.4}")
    print(f"m0_err = {m0_err:.4}")
    print("="*40)
    print(f"A0_mean = {A0_mean}")
    print(f"A0_err = {A0_err}")






    # data_fit,t_fit,yerr_fit = jackknife_fit(t,t_min=tmin,t_max=tmax,data=data)

    # out = one_state_fit(t_fit=t_fit,data_fit=data_fit,yerr_fit=yerr_fit,T=T)
    # # two_state_fit(t_fit=t_fit,data_fit=data_fit,yerr_fit=yerr_fit,T=T)

    



