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


def fit_func(params,t):
    A0 = params['A0']
    A1 = params['A1'] 
    m0 = params['m0']
    m1 = params['m1']
    return A0 * np.exp(-m0 * t) + A1 * np.exp(-m1 * t)


def residual(params,t_fit,data_fit,err_fit):
    model = fit_func(params,t_fit)
    return (data_fit - model)/err_fit


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


def two_state_fit(t_min,t_max,t,data):
    
    t_fit, data_fit,err_fit = jackknife_fit(t_min,t_max,t,data)
    # 基态
    params = Parameters()
    params.add('A0', value=0.01839, min=0)
    params.add('A1', value=0.012, min=0)
    params.add('m0', value=0.65073, min=0.1, max=1.5)        
    params.add('m1', value=0.8, min=0.6, max=3.0)

    result = minimize(residual, params, method='leastsq',
                           kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit})
    
    return t_fit,data_fit,err_fit,result



if __name__ == "__main__":

    data,t,N_cfg = data_load()

    t_fit,data_fit,err_fit,result = two_state_fit(t_min=5,t_max=16,t=t,data=data)

    print(fit_report(result))