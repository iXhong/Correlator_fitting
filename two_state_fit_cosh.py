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


def fit_func(params,T,t):
    A0 = params['A0']
    m0 = params['m0']
    A1 = params['A1']
    m1 = params['m1']

    return A0*np.cosh(m0*(t-T/2)) + A1*np.cosh(m1*(t-T/2))


def fit_func_cosh(params,T,t):
    A0 = params['A0']
    m0 = params['m0']

    return A0*np.cosh(m0*(t-T/2))


def jackknife_fit(t,t_min,t_max,data):
    def func(data):
        return np.mean(data,axis=0)
    
    N_cfg = data.shape[0]
    mask = (t>=t_min) & (t<=t_max)

    normalized_data = data / data[0,24]
 
    jk_mean,jk_err = jackknife(f=func,data=normalized_data,numb_blocks=N_cfg,return_sample=False,conf_axis=0)

    t_fit = t[mask]
    data_fit = jk_mean[mask]
    yerr_fit = jk_err[mask]

    return data_fit,t_fit,yerr_fit


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


def two_state_fit(t_fit,data_fit,yerr_fit,T):
    params = Parameters()
    params.add('A0', value=2.5541e-07, min=1e-7)
    params.add('m0', value=0.65073911, min=0.6, max=0.8)
    params.add('A1', value=1e-10,min=0,max=1e-7) 
    params.add('m1', value=1.0, min=0.7, max=5.0)

    out = minimize(residual,params=params,method='least_squares',kws={"data_fit":data_fit,"t_fit":t_fit,"yerr_fit":yerr_fit,"func":fit_func,"T":T})

    print(fit_report(out))
    return out


def out_plot(out,data,normalized):
    if not normalized:
        A0 = out.params['A0'].value
        A1 = out.params['A1'].value
        m0 = out.params['m0'].value
        m1 = out.params['m1'].value
    else:
        A0 = out.params['A0'].value * data[0,24]
        A1 = out.params['A1'].value * data[0,24]
        m0 = out.params['m0'].value
        m1 = out.params['m1'].value
 
    def two_state_cosh(t, A0, m0, A1, m1, T):
        """Two-state fit: ground state + first excited state"""
        return (A0 * np.cosh(m0 * (t - T/2)) + 
                A1 * np.cosh(m1 * (t - T/2)))
    
    y = two_state_cosh(t,A0,m0,A1,m1,T)

    plt.figure()
    plt.scatter(t,np.log(y))
    plt.scatter(t,np.log(np.abs(data[0,:])))
    plt.show()


if __name__ == "__main__":

    tmin,tmax = 5,16
    T=96

    data,t,N_cfg = data_load()
    data_fit,t_fit,yerr_fit = jackknife_fit(t,t_min=tmin,t_max=tmax,data=data)

    # out = one_state_fit(t_fit=t_fit,data_fit=data_fit,yerr_fit=yerr_fit,T=T)
    out = two_state_fit(t_fit=t_fit,data_fit=data_fit,yerr_fit=yerr_fit,T=T)

    out_plot(out,data,True)




    



