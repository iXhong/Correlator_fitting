import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from latqcdtools.statistics.jackknife import jackknife
from load_data import data_load


def func(data):
    return np.mean(data,axis=0)


def jk_mean_err(t_min,t_max,t,data):
    """
    利用jackknife方法获得用于拟合的samples和sigma
    Return:
        data_fit:jk samples to be fitted, shape (N_blocks, n_tfit)
        err_fit: SE in jk, used in WLS method.
    """
    N_blocks = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    scale_factor = 1e10

    data = data*scale_factor
    # 构造 jackknife samples
    jk_means = []
    for i in range(N_blocks):
        jk_indices = np.concatenate([np.arange(i), np.arange(i + 1, N_blocks)])
        jk_data_subset = data[jk_indices][:, fit_mask]
        jk_mean = np.mean(jk_data_subset, axis=0)
        jk_means.append(jk_mean)
    jk_means = np.array(jk_means)  # shape (N_blocks, n_tfit)

    # 计算全局误差
    jk_mean_overall = np.mean(jk_means, axis=0)
    diffs = jk_means - jk_mean_overall
    # jackknife 方差公式 (N-1)/N * sum (diff^2)
    sigma_global = np.sqrt((N_blocks - 1) / N_blocks * np.sum(diffs**2, axis=0))

    data_fit = jk_means
    err_fit = sigma_global
    
    return t_fit,data_fit,err_fit,scale_factor


def two_cosh_func(params,t,T):
    #function to fit
    A0 = params['A0']
    A1 = params['A1'] 
    m0 = params['m0']
    m1 = params['m1']  
    return A0*np.cosh(m0*(t-T/2)) + A1*np.cosh(m1*(t-T/2))


def double_cosh_residual(params, t_fit, data_fit, err_fit,T):
    #双指数残差函数
    model = two_cosh_func(params, t_fit,T)
    return (data_fit - model) / err_fit


def two_state_fit(t_fit,data_fit,err_fit,T,scale_factor):
    #双指数拟合
    params_double = Parameters()
    params_double.add('A0', value=1e-15*scale_factor, min=1e-16*scale_factor)
    params_double.add('A1', value=1e-17*scale_factor, min=0, max=1e-7*scale_factor)
    params_double.add('m0', value=0.60, min=0.6, max=0.8)
    params_double.add('m1', value=1.0, min=0.7, max=5.0)
    
    result = minimize(double_cosh_residual, params_double, method='least_squares',
                           kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit,"T":T})

    return result


def jackknife_fit(t_min,t_max,t,data,T):
    """
    uncorrelated 2 state jackknife fit
    """
    N_blocks = data.shape[0]
    #构造jackknife samples
    t_fit, data_fit, err_fit, scale_factor = jk_mean_err(t_min=t_min, t_max=t_max, t=t, data=data)

    #对每个jackknife sample拟合
    jk_params_A0 = []
    jk_params_m0 = []
    jk_params_A1 = []
    jk_params_m1 = []
    failed_fits = 0

    for i in range(N_blocks):
        result = two_state_fit(t_fit=t_fit,data_fit=data_fit[i,:],err_fit=err_fit,T=T,scale_factor=scale_factor)

        if result.success:
            jk_params_A0.append(result.params['A0'].value)
            jk_params_A1.append(result.params['A1'].value)
            jk_params_m0.append(result.params['m0'].value)
            jk_params_m1.append(result.params['m1'].value)
        else:
            failed_fits += 1
        
    if len(jk_params_A0) == 0:
        raise RuntimeError("All jackknife fit failed")
    
    #进行估计
    jk_params_A0 = np.array(jk_params_A0)/scale_factor #重新放缩
    jk_params_A1 = np.array(jk_params_A1)/scale_factor
    jk_params_m0 = np.array(jk_params_m0)
    jk_params_m1 = np.array(jk_params_m1)

    #计算mean和err
    A0_mean = np.mean(jk_params_A0)
    A0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_A0, ddof=0))

    m0_mean = np.mean(jk_params_m0)
    m0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_m0, ddof=0))

    A1_mean = np.mean(jk_params_A1)
    A1_std = np.sqrt((N_blocks - 1) * np.var(jk_params_A1, ddof=0))

    m1_mean = np.mean(jk_params_m1)
    m1_std = np.sqrt((N_blocks - 1) * np.var(jk_params_m1, ddof=0))

    #输出部分
    class JackknifeResult:
        def __init__(self,A0_mean,A0_std,A1_mean,A1_std,m0_mean,m0_std,m1_mean,m1_std,success_rate) -> None:
            self.params = {
                'A0': type("",(),{"value":A0_mean,'stderr':A0_std})(),
                'A1': type("",(),{"value":A1_mean,'stderr':A1_std})(),
                'm0': type("",(),{"value":m0_mean,'stderr':m0_std})(),
                'm1': type("",(),{"value":m1_mean,'stderr':m1_std})(),
            }
            self.success_rate = success_rate
            self.scale_factor = scale_factor

    return JackknifeResult(
        A0_mean,A0_std,A1_mean,A1_std,m0_mean,m0_std,m1_mean,m1_std,len(jk_params_A0)/N_blocks
    )


def out_plot(out, data, scale_factor, t_fit, t, T):
    #plot 
    A0 = out.params['A0'].value 
    A1 = out.params['A1'].value
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
    # plt.xlim(4,17)
    plt.xlabel('t')
    plt.ylabel('log(|C(t)|)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main(show_plot=True):
    data, t,_= data_load()
    T = 96

    out = jackknife_fit(t_min=5,t_max=16,t=t,data=data,T=T)

    print("="*50)
    print("拟合结果如下")
    print(f"A0= {out.params['A0'].value:.6e},err={out.params['A0'].stderr:.6e}")
    print(f"A1= {out.params['A1'].value:.6e},err={out.params['A1'].stderr:.6e}")
    print(f"m0= {out.params['m0'].value:.6f},err={out.params['m0'].stderr:.6e}")
    print(f"m1= {out.params['m1'].value:.6f},err={out.params['m1'].stderr:.6e}")

  

    if show_plot:  
        t_fit, data_fit, err_fit, scale_factor = jk_mean_err(t_min=5, t_max=16, t=t, data=data)
        out_plot(out, data, scale_factor, t_fit, t, T)  # 传入t_fit

    # print("="*50)
    # print("真实的拟合结果如下")
    # print(f"A0= {out.params['A0'].value / scale_factor:.6e}")
    # print(f"A1= {out.params['A1'].value / scale_factor:.6e}")
    # print(f"m0= {out.params['m0'].value:.6f}")
    # print(f"m1= {out.params['m1'].value:.6f}")


if __name__ == "__main__":

    main()
