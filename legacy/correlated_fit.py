"""
Name:correlated_fit
Author: George Liu
Since: 2025.9
Description: perform correlated fit on two point correlator 
"""


import numpy as np
import matplotlib.pyplot as plt
from latqcdtools.statistics.jackknife import jackknife
from latqcdtools.statistics.bootstr import bootstr
from lmfit import minimize,Parameters, fit_report
from load_data import data_load


def fit_function(params,t,T):
    A0 = params['A0']
    m0 = params['m0']

    return A0*np.cosh(m0*(t-T/2))


def jackknife_fit_with_covariance(t_min, t_max, t, data):
    """使用jackknife方法计算平均值、误差和协方差矩阵"""
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    
    # 计算平均值
    mean_corr = np.mean(data, axis=0)
    
    # 计算jackknife样本
    jk_samples = []
    for i in range(N_cfg):
        jk_sample = np.mean(np.delete(data, i, axis=0), axis=0)
        jk_samples.append(jk_sample)
    
    jk_samples = np.array(jk_samples)
    
    # 计算jackknife误差（对角线元素）
    jk_mean = np.mean(jk_samples, axis=0)
    jk_err = np.sqrt((N_cfg - 1) * np.mean((jk_samples - jk_mean)**2, axis=0))
    
    # 计算协方差矩阵
    # 对拟合范围内的数据计算协方差
    jk_samples_fit = jk_samples[:, fit_mask]
    jk_mean_fit = np.mean(jk_samples_fit, axis=0)
    
    # 协方差矩阵计算
    cov_matrix = np.zeros((len(t_fit), len(t_fit)))
    for i in range(N_cfg):
        diff = jk_samples_fit[i] - jk_mean_fit
        cov_matrix += np.outer(diff, diff)
    
    cov_matrix *= (N_cfg - 1) / N_cfg  # jackknife normalization
    
    data_fit = mean_corr[fit_mask]
    err_fit = jk_err[fit_mask]
    sigma = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(sigma,sigma)
    
    return t_fit, data_fit, err_fit, corr_matrix


def correlated_residual_improved(params, t_fit, data_fit, yerr_fit, corr_matrix,T):

    x = t_fit
    y = data_fit
    sigma = yerr_fit
    # model = model_func(params, x)
    model = fit_function(params,x,T)
    delta_norm = (y - model) / sigma
    
    # Compute inverse of correlation matrix
    inv_corr = np.linalg.inv(corr_matrix)
    
    # Cholesky decomposition of inv_corr: L such that L @ L.T = inv_corr
    L = np.linalg.cholesky(inv_corr)
    
    # Residuals: L.T @ delta_norm so that res.T @ res = delta_norm.T @ inv_corr @ delta_norm
    res = np.dot(L.T, delta_norm)
    
    return res
    

def correlated_fit(t_min,t_max,t,data,T):
    t_fit, data_fit, err_fit, corr_matrix = jackknife_fit_with_covariance(t_min=t_min,t_max=t_max,t=t,data=data)

    params = Parameters()
    params.add('A0',value=2.2423139998928857e-15,min=0)
    params.add('m0',value=0.6548,min=0)

    # out = minimize(correlated_residual,params=params,method='least_square',kws={"t_fit":t_fit,"data_fit":data_fit,"cov_matrix":cov_matrix,"T":T})

    out = minimize(correlated_residual_improved,params=params,method='least_squares',kws={"t_fit":t_fit,"data_fit":data_fit,"yerr_fit":err_fit,"corr_matrix":corr_matrix,"T":T})

    return out


def mean_func(data):
    return np.mean(data,axis=0)


if __name__ == "__main__":

    t_min,t_max = 5,16
    T=96

    data,t,N_cfg = data_load()
    out = correlated_fit(t_min=t_min,t_max=t_max,t=t,T=T,data=data)

    print(fit_report(out))

    A0 = out.params['A0'].value
    m0 = out.params['m0'].value

    y = A0*np.cosh(m0*(t-T/2))

    plt.figure()
    plt.scatter(t,np.log(y))
    plt.scatter(t,np.log(np.abs(data[0,:])))
    plt.show()




    



