import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from latqcdtools.statistics.jackknife import jackknife
from load_data import data_load


def fit_function(params, t):
    """使用单指数函数"""
    A0 = params['A0']
    m0 = params['m0']
    return A0 * np.exp(-m0 * t)

def fit_function_cosh(params,t,T):
    A0 = params['A0']
    m0 = params['m0']
    # T = params['T']
    return A0*np.cosh(m0*(t-T/2))

def jk_mean_err(t_min,t_max,t,data):
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]

    def func(data):
        return np.mean(data,axis=0)

    jk_mean,jk_err = jackknife(f=func,data=data,numb_blocks=N_cfg,conf_axis=0)

    data_fit = jk_mean[fit_mask]
    err_fit = jk_err[fit_mask]
    
    return t_fit, data_fit, err_fit

def weighted_residual(params, t_fit, data_fit, err_fit,T):
    """带权重的残差函数"""
    model = fit_function_cosh(params, t_fit,T)
    return (data_fit - model) / err_fit  # 用误差加权

def direct_fit(t_min, t_max, t, data,T,print_report):
    """改进的拟合函数，考虑统计误差"""
    t_fit, data_fit, err_fit = jk_mean_err(t_min, t_max, t, data)
    
    # print(f"拟合数据范围: {data_fit.min():.2e} 到 {data_fit.max():.2e}")
    # print(f"典型误差: {np.mean(err_fit):.2e}")
    
    params = Parameters()
    params.add('A0', value=data_fit[0], min=0)  
    params.add('m0', value=0.5, min=0)
    # params.add('T',value=96,vary=False)
    
    # 使用带权重的拟合
    result = minimize(weighted_residual, params, method='least_squares', 
                     kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit,"T":T})
    
    if print_report:
        print("="*50)
        print("带误差权重的拟合结果:")
        print(fit_report(result))
    
    return result, t_fit, data_fit, err_fit

def plot_fit_result(t, mean_corr, jk_err, result, t_fit, data_fit, err_fit,T):
    """绘制带误差棒的拟合结果"""
    plt.figure(figsize=(12, 8))
    
    # 绘制所有数据点（带误差棒）
    plt.errorbar(t, mean_corr, yerr=jk_err, fmt='bo', label='Data', 
                markersize=4, capsize=3, alpha=0.7)
    
    # 高亮拟合范围
    plt.errorbar(t_fit, data_fit, yerr=err_fit, fmt='ro', label='Fit range', 
                markersize=6, capsize=3)
    
    # 拟合曲线
    t_smooth = np.linspace(t_fit[0], t_fit[-1], 100)
    fit_curve = fit_function_cosh(result.params, t_smooth,T)
    plt.plot(t_smooth, fit_curve, 'r-', label='Fit', linewidth=2)
    
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('G(t)')
    plt.title('Improved Direct Fit with Error Bars')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 拟合质量检查
    chi2_per_dof = result.redchi
    print(f"\n拟合质量评估:")
    print(f"χ²/dof = {chi2_per_dof:.3f}")
    if chi2_per_dof < 0.1:
        print("⚠️  χ²/dof过小，可能过拟合或误差估计过大")
    elif chi2_per_dof > 2.0:
        print("⚠️  χ²/dof过大，拟合质量较差")
    else:
        print("✅ χ²/dof合理，拟合质量良好")


def plot_result(result,mean_corr,t,T):
    A0 = result.params['A0'].value
    m0 = result.params['m0'].value

    y =  A0 * np.cosh(m0*(t-T/2))
    # print(y[5:16])
    # print(mean_corr[5:16])

    plt.figure()
    plt.scatter(t,np.log(y))
    plt.scatter(t,np.log(np.abs(mean_corr)))
    plt.show()


def main(show_plot,print_report):
    data, t, N_cfg = data_load()
    mean_corr = np.mean(data, axis=0)
    T = 96

    result, t_fit, data_fit, err_fit = direct_fit(t_min=5, t_max=48, t=t, data=data,T=T,print_report=print_report)
    
    if show_plot:
        # plot_fit_result(t, mean_corr, result, t_fit, data_fit)
        # plot_fit_result(t=t,mean_corr=mean_corr,jk_err=)
        plot_result(result,mean_corr,t,T)

    return {
        "result":result,
        "t_fit":t_fit,
        "data_fit":data_fit,
        "err_fit":err_fit
    }


if __name__ == "__main__":

    out = main(show_plot=False,print_report=True)
    # filp_data,t,N = data_load()
    # print(filp_data.shape)
