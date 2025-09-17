import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
from latqcdtools.statistics.jackknife import jackknife
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

def residual(params, t_fit, data_fit):
    """简单的残差函数，不使用协方差矩阵加权"""
    model = fit_function(params, t_fit)
    return data_fit - model


def fit_function(params, t):
    """使用单指数函数"""
    A0 = params['A0']
    m0 = params['m0']
    return A0 * np.exp(-m0 * t)


def jackknife_fit(t_min, t_max, t, data):
    """使用jackknife方法进行拟合"""
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    
    # 计算平均值和jackknife误差
    mean_corr = np.mean(data, axis=0)
    jk_samples = []
    
    for i in range(N_cfg):
        # 删除第i个配置的jackknife样本
        jk_sample = np.mean(np.delete(data, i, axis=0), axis=0)
        jk_samples.append(jk_sample)
    
    jk_samples = np.array(jk_samples)
    
    # 计算jackknife误差
    jk_mean = np.mean(jk_samples, axis=0)
    jk_err = np.sqrt((N_cfg - 1) * np.mean((jk_samples - jk_mean)**2, axis=0))
    
    data_fit = mean_corr[fit_mask]
    err_fit = jk_err[fit_mask]
    
    return t_fit, data_fit, err_fit

def weighted_residual(params, t_fit, data_fit, err_fit):
    """带权重的残差函数"""
    model = fit_function(params, t_fit)
    return (data_fit - model) / err_fit  # 用误差加权

def improved_direct_fit(t_min, t_max, t, data):
    """改进的拟合函数，考虑统计误差"""
    t_fit, data_fit, err_fit = jackknife_fit(t_min, t_max, t, data)
    
    print(f"拟合数据范围: {data_fit.min():.2e} 到 {data_fit.max():.2e}")
    print(f"典型误差: {np.mean(err_fit):.2e}")
    
    params = Parameters()
    params.add('A0', value=data_fit[0], min=0)  
    params.add('m0', value=0.5, min=0.1, max=2.0)
    params.add('T',value=96,vary=False)
    
    # 使用带权重的拟合
    result = minimize(weighted_residual, params, method='leastsq', 
                     kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit})
    
    print("="*50)
    print("带误差权重的拟合结果:")
    
    return result, t_fit, data_fit, err_fit


def test_different_fit_ranges(t, data):
    """测试不同拟合范围的效果"""
    ranges_to_test = [
        (5, 16),
        (6, 16),   # 更宽范围
        (8, 16),   # 当前右端点扩展
        (6, 14),   # 当前左端点扩展
        (8, 14),   # 当前范围
    ]
    
    results = []
    for t_min, t_max in ranges_to_test:
        print(f"\n测试拟合范围: t = {t_min} 到 {t_max}")
        try:
            result, t_fit, data_fit, err_fit = improved_direct_fit(t_min, t_max, t, data)
            results.append({
                'range': (t_min, t_max),
                'n_points': len(t_fit),
                'chi2_dof': result.redchi,
                'm0': result.params['m0'].value,
                'm0_err': result.params['m0'].stderr,
                'A0': result.params['A0'].value,
            })
        except Exception as e:
            print(f"拟合失败: {e}")
    
    # 比较结果
    print("\n" + "="*80)
    print("不同拟合范围的结果比较:")
    print("范围\t\t点数\tχ²/dof\t\tm0\t\tm0误差\t\tA0")
    for r in results:
        print(f"{r['range']}\t\t{r['n_points']}\t{r['chi2_dof']:.3f}\t\t"
              f"{r['m0']:.4f}\t\t{r['m0_err']:.5f}\t\t{r['A0']:.5f}")


def plot_improved_fit_result(t, mean_corr, jk_err, result, t_fit, data_fit, err_fit):
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
    fit_curve = fit_function(result.params, t_smooth)
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


def plot_result(result,mean_corr):
    A0 = result.params['A0'].value
    m0 = result.params['m0'].value

    y =  A0 * np.exp(-m0*t)

    plt.figure()
    plt.scatter(t,np.log(y))
    plt.scatter(t,np.log(np.abs(mean_corr)))
    plt.show()


if __name__ == "__main__":
    data, t, N_cfg = data_load()
    mean_corr = np.mean(data, axis=0)
    
    # 使用改进的拟合方法
    result, t_fit, data_fit, err_fit = improved_direct_fit(t_min=5, t_max=16, t=t, data=data)
    print(fit_report(result))
    
    # 计算jackknife误差用于绘图
    _, _, jk_err = jackknife_fit(0, len(t)-1, t, data)
    
    # 绘制改进的结果
    plot_improved_fit_result(t, mean_corr, jk_err, result, t_fit, data_fit, err_fit)
    
    # 绘制结果
    # plot_fit_result(t, mean_corr, result, t_fit, data_fit)
    plot_result(result,mean_corr)

    # 调用测试
    # test_different_fit_ranges(t, data)