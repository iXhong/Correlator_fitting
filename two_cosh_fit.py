import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from latqcdtools.statistics.jackknife import jackknife
from load_data import data_load


def func(data):
    return np.mean(data,axis=0)


def jk_mean_err(t_min,t_max,t,data):
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]

    #scale the data
    scale_factor = 1e10
    data = data * scale_factor
    jk_mean, jk_err = jackknife(f=func,data=data,numb_blocks=N_cfg,conf_axis=0,return_sample=False)

    data_fit = jk_mean[fit_mask]
    err_fit = jk_err[fit_mask]
    
    return t_fit,data_fit,err_fit,scale_factor


def two_cosh_func(params,t,T):
    A0 = params['A0']
    A1 = params['A1'] 
    m0 = params['m0']
    m1 = params['m1']  
    return A0*np.cosh(m0*(t-T/2)) + A1*np.cosh(m1*(t-T/2))


def double_cosh_residual(params, t_fit, data_fit, err_fit,T):
    """双指数残差函数"""
    model = two_cosh_func(params, t_fit,T)
    return (data_fit - model) / err_fit


def two_state_fit(t_fit,data_fit,err_fit,T,scale_factor):
    # 2. 双指数拟合
    params_double = Parameters()
    # 初始参数设置很重要
    params_double.add('A0', value=1e-15*scale_factor, min=1e-16*scale_factor)
    params_double.add('A1', value=1e-17*scale_factor, min=0, max=1e-7*scale_factor)
    # params_double.add('m0', value=0.60073944, min=0.6, max=1.5)        # 基态
    # params_double.add('m1', value=1.0, min=0.7, max=3.0)        # 激发态，应该更大
    params_double.add('m0', value=0.60, min=0.6, max=0.8)  # ⚠️ 更严格的上界
    params_double.add('m1', value=1.0, min=0.7, max=5.0)
    
    result_double = minimize(double_cosh_residual, params_double, method='least_squares',
                           kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit,"T":T})

    return result_double

def out_plot(out, data, scale_factor, t_fit):
    A0 = out.params['A0'].value / scale_factor
    A1 = out.params['A1'].value / scale_factor
    m0 = out.params['m0'].value
    m1 = out.params['m1'].value
 
    def two_state_cosh(t, A0, m0, A1, m1, T):
        return (A0 * np.cosh(m0 * (t - T/2)) + 
                A1 * np.cosh(m1 * (t - T/2)))
    
    y_fit = two_state_cosh(t_fit, A0, m0, A1, m1, T)
    y_all = two_state_cosh(t, A0, m0, A1, m1, T)

    plt.figure(figsize=(10, 6))
    plt.errorbar(t, np.log(np.abs(data[0, :])), fmt='o', label='Data', alpha=0.7)
    plt.plot(t, np.log(y_all), 'r-', label='Fit (full range)', alpha=0.8)
    plt.plot(t_fit, np.log(y_fit), 'g-', linewidth=3, label='Fit (fit range)')
    plt.xlabel('t')
    plt.ylabel('log(|C(t)|)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    data, t, N_cfg = data_load()
    T = 96

    t_fit, data_fit, err_fit, scale_factor = jk_mean_err(t_min=5, t_max=16, t=t, data=data)

    out = two_state_fit(t_fit=t_fit, data_fit=data_fit, err_fit=err_fit, T=T, scale_factor=scale_factor)
    print(fit_report(out))

    out_plot(out, data, scale_factor, t_fit)  # 传入t_fit

    print("="*50)
    print("真实的拟合结果如下")
    print(f"A0= {out.params['A0'].value / scale_factor:.6e}")
    print(f"A1= {out.params['A1'].value / scale_factor:.6e}")
    print(f"m0= {out.params['m0'].value:.6f}")
    print(f"m1= {out.params['m1'].value:.6f}")

# if __name__ == "__main__":
#     data, t, N_cfg = data_load()
#     T = 96
#     t_min, t_max = 5, 16
    
#     # 详细检查拟合范围
#     fit_mask = (t >= t_min) & (t <= t_max)
#     t_fit_check = t[fit_mask]
#     print(f"时间数组 t: {t}")
#     print(f"拟合掩码: {fit_mask}")
#     print(f"拟合时间点: {t_fit_check}")
#     print(f"拟合点数: {len(t_fit_check)}")
    
#     t_fit, data_fit, err_fit, scale_factor = jk_mean_err(t_min=t_min, t_max=t_max, t=t, data=data)
    
#     print(f"实际拟合点数: {len(t_fit)}")
#     print(f"实际拟合时间点: {t_fit}")
    
#     out = two_state_fit(t_fit=t_fit, data_fit=data_fit, err_fit=err_fit, T=T, scale_factor=scale_factor)
#     print(fit_report(out))

#     print("="*50)
#     print("真实的拟合结果如下")
#     print(f"A0= {out.params['A0'].value / scale_factor:.6e}")
#     print(f"A1= {out.params['A1'].value / scale_factor:.6e}")
#     print(f"m0= {out.params['m0'].value:.6f}")
#     print(f"m1= {out.params['m1'].value:.6f}")