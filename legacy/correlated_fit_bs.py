import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize,Parameters, fit_report
from load_data import data_load


def fit_function(params,t,T):
    A0 = params['A0']
    m0 = params['m0']

    return A0*np.cosh(m0*(t-T/2))   


def bootstrap_fit_with_covariance(t_min, t_max, t, data, n_bootstrap=1000):
    """使用bootstrap方法计算平均值、误差和协方差矩阵"""
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    
    # 计算平均值
    mean_corr = np.mean(data, axis=0)
    
    # 计算bootstrap样本
    bs_samples = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, N_cfg, N_cfg)
        bs_sample = np.mean(data[indices], axis=0)
        bs_samples.append(bs_sample)
    
    bs_samples = np.array(bs_samples)
    
    # 计算bootstrap误差（对角线元素）
    bs_mean = np.mean(bs_samples, axis=0)
    bs_err = np.std(bs_samples, axis=0)
    
    # 计算协方差矩阵
    # 对拟合范围内的数据计算协方差
    bs_samples_fit = bs_samples[:, fit_mask]
    bs_mean_fit = np.mean(bs_samples_fit, axis=0)
    
    # 协方差矩阵计算
    cov_matrix = np.zeros((len(t_fit), len(t_fit)))
    for i in range(n_bootstrap):
        diff = bs_samples_fit[i] - bs_mean_fit
        cov_matrix += np.outer(diff, diff)
    
    cov_matrix /= n_bootstrap  # bootstrap normalization
    
    data_fit = mean_corr[fit_mask]
    err_fit = bs_err[fit_mask]
    sigma = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(sigma,sigma)
    
    return t_fit, data_fit, err_fit, corr_matrix

def correlated_residual(params,t_fit,data_fit,yerr_fit,corr_matrix,T):
    """计算考虑协方差矩阵的相关残差"""
    model = fit_function(params,t_fit,T)
    residual = data_fit - model
    # 使用协方差矩阵计算加权残差
    inv_corr = np.linalg.inv(corr_matrix)
    weighted_residual = np.dot(inv_corr, residual / yerr_fit)
    return weighted_residual


def correlated_fit(t_min,t_max,t,data,T):
    
    t_fit, data_fit, err_fit, corr_matrix = bootstrap_fit_with_covariance(t_min, t_max, t, data, n_bootstrap=100)
    # 设置拟合参数
    params = Parameters()
    params.add('A0',value=2.2423139998928857e-15,min=0)
    params.add('m0',value=0.6548,min=0)

    result = minimize(correlated_residual,params=params,method='least_squares',kws={"t_fit":t_fit,"data_fit":data_fit,"yerr_fit":err_fit,"corr_matrix":corr_matrix,"T":T})

    return result



if __name__ == "__main__":

    t_min,t_max = 5,16
    T=96

    data,t,N_cfg = data_load()

    out = correlated_fit(t_min,t_max,t,data,T)
    print(fit_report(out))

    A0 = out.params['A0'].value
    m0 = out.params['m0'].value

    y = A0*np.cosh(m0*(t-T/2))

    plt.figure()
    plt.scatter(t,np.log(y))
    plt.scatter(t,np.log(np.abs(data[0,:])))
    plt.show()