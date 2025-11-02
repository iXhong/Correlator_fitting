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

def direct_fit(t_min, t_max, t, data):
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    data_fit = data[fit_mask]
    
    # 检查数据
    print(f"拟合数据范围: {data_fit.min():.2e} 到 {data_fit.max():.2e}")
    
    params = Parameters()
    params.add('A0', value=data_fit[0], min=0)  
    params.add('m0', value=0.5, min=0.1, max=2.0)
    
    result = minimize(residual, params, method='leastsq', 
                     kws={"t_fit": t_fit, "data_fit": data_fit})
    
    return result, t_fit, data_fit

def plot_fit_result(t, data, result, t_fit, data_fit):
    """绘制拟合结果"""
    plt.figure(figsize=(10, 6))
    
    # 绘制原始数据
    plt.semilogy(t, data, 'bo', label='Data', markersize=4)
    
    # 绘制拟合范围的数据（高亮显示）
    plt.semilogy(t_fit, data_fit, 'ro', label='Fit range', markersize=6)
    
    # 绘制拟合曲线
    t_smooth = np.linspace(t_fit[0], t_fit[-1], 100)
    fit_curve = fit_function(result.params, t_smooth)
    plt.semilogy(t_smooth, fit_curve, 'r-', label='Fit', linewidth=2)
    
    plt.xlabel('t')
    plt.ylabel('G(t)')
    plt.title('Direct Fit Result')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def check_data_quality(t, data):
    """检查数据质量"""
    print(f"数据范围: {data.min():.2e} 到 {data.max():.2e}")
    print(f"数据是否包含负值: {np.any(data < 0)}")
    print(f"数据是否包含零值: {np.any(data == 0)}")
    print(f"数据是否单调递减: {np.all(np.diff(data) <= 0)}")
    
    # 绘制数据来检查
    plt.figure(figsize=(8, 6))
    plt.semilogy(t, data, 'bo-')
    plt.xlabel('t')
    plt.ylabel('G(t)')
    plt.title('data check')
    plt.grid(True)
    plt.show()


def plot_result(result,mean_corr):
    A0 = result.params['A0'].value
    m0 = result.params['m0'].value

    y =  A0 * np.exp(-m0*t)

    plt.figure()
    plt.scatter(t,np.log(y))
    plt.scatter(t,np.log(np.abs(mean_corr)))
    plt.show()


if __name__ == "__main__":
    # 加载数据
    data, t, N_cfg = data_load()
    # 计算平均值（直接拟合只需要平均值）
    mean_corr = np.mean(data, axis=0)
    # 在main中调用
    # check_data_quality(t, mean_corr)


    print("数据加载完成")
    # print(f"数据形状: {data.shape}")
    # print(f"平均关联子前几个值: {mean_corr[:5]}")
    
    # 进行直接拟合
    result, t_fit, data_fit = direct_fit(t_min=5, t_max=16, t=t, data=mean_corr)
    print(fit_report(result))
    
    # 绘制结果
    plot_fit_result(t, mean_corr, result, t_fit, data_fit)
    plot_result(result,mean_corr)