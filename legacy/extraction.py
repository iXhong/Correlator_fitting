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
    t = np.array(range(C_array.shape[1]))

    data = C_array

    N_cfg = data.shape[0] # configuration num
    flip_data = (data[:,:48] + np.flip(data[:,-48:]))/2 #filp & average data
    t = np.arange(flip_data.shape[1])
    print(f'{N_cfg} 个组态，{len(t)} 个时间点')

    return flip_data,t,N_cfg


def jackknife_resample(data):
    def func(data):
        return np.mean(data,axis=0)
    jk_samples,jk_mean,jk_err = jackknife(func,data,numb_blocks=N_cfg,conf_axis=0,return_sample=True)

    return jk_samples


def get_cov_matrix(jk_samples,mean_corr):
    T = jk_samples.shape[1]
    N_cfg = jk_samples.shape[0]

    cov_matrix = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            diff_i = jk_samples[:, i] - mean_corr[i]
            diff_j = jk_samples[:, j] - mean_corr[j]
            cov_matrix[i, j] = (N_cfg - 1) / N_cfg * np.sum(diff_i * diff_j)
    
    return cov_matrix


"""
有效质量
"""
def effective_mass(mean_corr,T):
    m_eff = np.zeros(T - 1)
    for t in range(T - 1):
        if mean_corr[t] > 0 and mean_corr[t+1] > 0:
            ratio = mean_corr[t] / mean_corr[t+1]
            if ratio > 1:
                m_eff[t] = np.log(ratio)
            else:
                m_eff[t] = np.nan
        else:
            m_eff[t] = np.nan
    
    return m_eff


def fit_function(params, t):
    """拟合函数: A0 * cosh(-m0 * (t - T/2))"""
    A0 = params['A0']
    m0 = params['m0']
    T = params['T']
    return A0 * np.cosh(m0 * (t - T/2))

def residual(params, t_fit, data_fit, inv_cov):
    """残差函数，用于lmfit minimize"""
    model = fit_function(params, t_fit)
    residual_vec = data_fit - model
    
    # 方法1：返回加权残差向量（推荐）
    # 对协方差矩阵进行Cholesky分解
    try:
        L = np.linalg.cholesky(inv_cov)
        weighted_residual = L @ residual_vec
        return weighted_residual
    except np.linalg.LinAlgError:
        # 如果Cholesky分解失败，使用对角加权
        weights = np.sqrt(np.diag(inv_cov))
        return weights * residual_vec


def one_state_fit(t_min,t_max,t,data,cov_matrix):
    '''单态拟合'''
    fit_mask = (t>=t_min) & (t<=t_max)
    t_fit = t[fit_mask]
    data_fit = data[fit_mask]
    cov_fit = cov_matrix[np.ix_(fit_mask, fit_mask)]

    # 正则化协方差矩阵
    reg_param = 1e-10 * np.trace(cov_fit) / len(cov_fit)
    cov_fit += reg_param * np.eye(len(cov_fit))

    try:
        inv_cov = np.linalg.inv(cov_fit)
    except np.linalg.LinAlgError:
        # 如果协方差矩阵奇异，使用对角近似
        inv_cov = np.diag(1.0 / np.diag(cov_fit))
    
    # 设置初始参数
    params = Parameters()
    params.add('A0', value=data_fit[0], min=0)
    params.add('m0', value=0.6, min=0.01, max=2.0)
    params.add('T', value=96, vary=False)

    # 进行拟合
    out = minimize(residual, params, method='leastsq', 
                  kws={"t_fit": t_fit, "data_fit": data_fit, "inv_cov": inv_cov})

    print(fit_report(out))
    return out


def plot_G(t, mean_corr, err_corr):
    plt.figure()
    plt.errorbar(t, mean_corr, err_corr, fmt='o', capsize=3)
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('G(t)')
    plt.title('Correlator')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_eff_mass(t, m_eff):
    valid_mask = ~np.isnan(m_eff)
    # 自动处理时间数组长度
    if len(t) == len(m_eff) + 1:
        t_plot = t[1:]
    else:
        t_plot = t[:len(m_eff)]
    
    plt.figure()
    plt.plot(t_plot[valid_mask], m_eff[valid_mask], 'o-', label='m_eff')
    plt.xlabel('t')
    plt.ylabel('m_eff(t)')
    plt.title('Effective Mass')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    data, t, N_cfg = data_load()
    mean_corr = np.mean(data, axis=0)
    std_corr = np.std(data, axis=0, ddof=1)
    err_corr = std_corr / np.sqrt(N_cfg)

    jk_samples = jackknife_resample(data)
    T = 96
    
    # 绘图
    # plot_G(t, mean_corr, err_corr)
    # m_eff = effective_mass(mean_corr, T)
    # plot_eff_mass(t, m_eff)

    cov_matrix = get_cov_matrix(jk_samples, mean_corr)
    
    # 进行拟合
    result = one_state_fit(9, 15, t, mean_corr, cov_matrix)

    # plt.figure()

    A0 = result.params['A0'].value
    m0 = result.params['m0'].value

    y =  A0 * np.cosh(-m0 * (t - T/2))

    plt.figure()
    plt.scatter(t,np.log(y))
    plt.scatter(t,np.log(np.abs(mean_corr)))
    plt.show()
    

