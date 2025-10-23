import numpy as np
import matplotlib.pyplot as plt
from latqcdtools.statistics.jackknife import jackknife
from lmfit import minimize,Parameters, fit_report
from load_data import data_load


def func(data):
    return np.mean(data,axis=0)

def fit_func(params,t,T):
    A0 = params['A0']
    A1 = params['A1'] 
    m0 = params['m0']
    m1 = params['m1']  
    return A0*np.cosh(m0*(t-T/2)) + A1*np.cosh(m1*(t-T/2))

def jackknife_fit(t_min,t_max,t,data):  
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]

    scale_factor = 1e10
    data = data * scale_factor
    jk_mean, jk_err = jackknife(f=func,data=data,numb_blocks=N_cfg,conf_axis=0,return_sample=False)

    data_fit = jk_mean[fit_mask]
    err_fit = jk_err[fit_mask]
    
    return t_fit,data_fit,err_fit,scale_factor

def residual(params, t_fit, data_fit, err_fit,T):
    model = fit_func(params, t_fit,T)
    return (data_fit - model) / err_fit

def two_state_fit(t_fit,data_fit,err_fit,T,scale_factor):
    params = Parameters()
    params.add('A0', value=1e-15*scale_factor, min=1e-16*scale_factor)
    params.add('A1', value=1e-17*scale_factor, min=0, max=1e-7*scale_factor)
    params.add('m0', value=0.60, min=0.6, max=0.8)
    params.add('m1', value=1.0, min=0.7, max=5.0)
    
    result = minimize(residual, params, method='least_squares',
                     kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit,"T":T})

    return result


if __name__ == "__main__":
    data, t, N_cfg = data_load()
    T = 96

    t_fit, data_fit, err_fit, scale_factor = jackknife_fit(t_min=5, t_max=16, t=t, data=data) 

    # out = two_state_fit(t_fit=t_fit, data_fit=data_fit, err_fit=err_fit, T=T, scale_factor=scale_factor)
    # print(fit_report(out))

    # print("="*50)
    # print("真实的拟合结果如下")
    # print(f"A0= {out.params['A0'].value / scale_factor:.6e}")
    # print(f"A1= {out.params['A1'].value / scale_factor:.6e}")
    # print(f"m0= {out.params['m0'].value:.6f}")
    # print(f"m1= {out.params['m1'].value:.6f}")

        # 新增的稳定性分析
    t_min_list, m0_values, m0_errors = analyze_m0_stability(
        data=data, t=t, N_cfg=N_cfg, T=T, 
        t_max=16, t_min_range=(3, 12)
    )

    # plt.figure()
    # plt.errorbar(t_min_list,m0_values,m0_errors,fmt='o')
    # plt.ylim((0.6,0.65))
    # plt.show()

    # 绘制稳定性图
    plot_m0_stability(t_min_list, m0_values, m0_errors, t_max=16)


    



