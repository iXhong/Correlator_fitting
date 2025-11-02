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


def fit_function_cosh(params,t,T):
    A0 = params['A0']
    m0 = params['m0']
    # T = params['T']
    return A0*np.cosh(m0*(t-T/2))


def residual(params, t_fit, data_fit, err_fit,T):
    """带权重的残差函数"""
    model = fit_function_cosh(params, t_fit,T)
    return (data_fit - model) / err_fit  # 用误差加权import numpy as np

def jackknife_fit(t_min, t_max, t, data, T,print_report):
    """
    改进版本：更清晰地处理sigma
    """
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    
    # 预先计算所有jackknife样本的统计量
    jk_means = []
    jk_sigmas = []
    failed_fits = 0
    
    for i in range(N_cfg):
        jk_indices = np.concatenate([np.arange(i), np.arange(i+1, N_cfg)])
        jk_data_subset = data[jk_indices][:, fit_mask]  # 只取拟合范围
        
        jk_mean = np.mean(jk_data_subset, axis=0)
        jk_std = np.std(jk_data_subset, axis=0, ddof=1)
        jk_sigma = jk_std / np.sqrt(N_cfg-1)  # 这就是每个观测点的sigma
        
        jk_means.append(jk_mean)
        jk_sigmas.append(jk_sigma)
    
    # jk_means = np.array(jk_means)  # shape: (N_cfg, len(t_fit))
    # jk_sigmas = np.array(jk_sigmas)  # shape: (N_cfg, len(t_fit))


    # 对每个jackknife样本进行拟合

    jk_params_A0 = []
    jk_params_m0 = []
    for i in range(N_cfg):
        jk_data_fit = jk_means[i]  # 观测值
        jk_err_fit = jk_sigmas[i]  # 观测误差（sigma）
        
        # 拟合：使用 (y_data - model) / sigma 作为残差
        # 这样权重就是 1/sigma^2
        try:
            # 设置参数
            params = Parameters()
            params.add('A0', value=jk_data_fit[0], min=0)  
            params.add('m0', value=0.5, min=0)
            
            # 执行拟合
            result = minimize(residual, params, method='least_squares', 
                            kws={"t_fit": t_fit, "data_fit": jk_data_fit, 
                                 "err_fit": jk_err_fit, "T": T})
            
            if result.success:
                jk_params_A0.append(result.params['A0'].value)
                jk_params_m0.append(result.params['m0'].value)
            else:
                failed_fits += 1
                
        except Exception:
            failed_fits += 1
            continue
    
    if len(jk_params_A0) == 0:
        raise RuntimeError("所有jackknife拟合都失败了")
    
    # 计算jackknife统计量
    jk_params_A0 = np.array(jk_params_A0)
    jk_params_m0 = np.array(jk_params_m0)
    
    A0_mean = np.mean(jk_params_A0)
    A0_std = np.sqrt((N_cfg - 1) * np.var(jk_params_A0, ddof=0))
    
    m0_mean = np.mean(jk_params_m0)
    m0_std = np.sqrt((N_cfg - 1) * np.var(jk_params_m0, ddof=0))

        # 创建一个类似lmfit结果的对象
    class JackknifeResult:
        def __init__(self, A0_mean, A0_std, m0_mean, m0_std, success_rate):
            self.params = {
                'A0': type('', (), {'value': A0_mean, 'stderr': A0_std})(),
                'm0': type('', (), {'value': m0_mean, 'stderr': m0_std})()
            }
            self.success = True
            self.chisqr = None  # 可以后续计算
            self.success_rate = success_rate

    if print_report:
        print("="*50)
        print("Jackknife拟合结果:")
        print(f"成功率: {len(jk_params_A0)}/{N_cfg} ({100*len(jk_params_A0)/N_cfg:.1f}%)")
        print(f"A0: {A0_mean} ± {A0_std}")
        print(f"m0: {m0_mean:.6f} ± {m0_std:.6f}")
    
    result = JackknifeResult(A0_mean, A0_std, m0_mean, m0_std, 
                            len(jk_params_A0)/N_cfg)
    
    return result


def jackknife_fit_v2(t_min, t_max, t, data, T, print_report):
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    
    # 使用全数据计算权重（推荐）
    full_data_fit = data[:, fit_mask]
    full_mean = np.mean(full_data_fit, axis=0)
    full_std = np.std(full_data_fit, axis=0, ddof=1)
    full_sigma = full_std / np.sqrt(N_cfg)  # 全数据的标准误
    
    jk_params_A0 = []
    jk_params_m0 = []
    
    for i in range(N_cfg):
        # 创建jackknife子集
        jk_indices = np.concatenate([np.arange(i), np.arange(i+1, N_cfg)])
        jk_data_subset = data[jk_indices][:, fit_mask]
        jk_mean = np.mean(jk_data_subset, axis=0)
        
        # 使用全数据的sigma作为权重（关键改进）
        try:
            params = Parameters()
            params.add('A0', value=jk_mean[0], min=0)
            params.add('m0', value=0.5, min=0)
            
            result = minimize(residual, params, method='least_squares',
                            kws={"t_fit": t_fit, "data_fit": jk_mean,
                                 "err_fit": full_sigma, "T": T})  # 注意这里用full_sigma
            
            if result.success:
                jk_params_A0.append(result.params['A0'].value)
                jk_params_m0.append(result.params['m0'].value)
        except Exception:
            continue
    
        
    if len(jk_params_A0) == 0:
        raise RuntimeError("所有jackknife拟合都失败了")
    
    # 计算jackknife统计量
    jk_params_A0 = np.array(jk_params_A0)
    jk_params_m0 = np.array(jk_params_m0)
    
    A0_mean = np.mean(jk_params_A0)
    A0_std = np.sqrt((N_cfg - 1) * np.var(jk_params_A0, ddof=0))
    
    m0_mean = np.mean(jk_params_m0)
    m0_std = np.sqrt((N_cfg - 1) * np.var(jk_params_m0, ddof=0))

        # 创建一个类似lmfit结果的对象
    class JackknifeResult:
        def __init__(self, A0_mean, A0_std, m0_mean, m0_std, success_rate):
            self.params = {
                'A0': type('', (), {'value': A0_mean, 'stderr': A0_std})(),
                'm0': type('', (), {'value': m0_mean, 'stderr': m0_std})()
            }
            self.success = True
            self.chisqr = None  # 可以后续计算
            self.success_rate = success_rate

    if print_report:
        print("="*50)
        print("Jackknife拟合结果:")
        print(f"成功率: {len(jk_params_A0)}/{N_cfg} ({100*len(jk_params_A0)/N_cfg:.1f}%)")
        print(f"A0: {A0_mean} ± {A0_std}")
        print(f"m0: {m0_mean:.6f} ± {m0_std:.6f}")
    
    result = JackknifeResult(A0_mean, A0_std, m0_mean, m0_std, 
                            len(jk_params_A0)/N_cfg)
    
    return result

def jackknife_fit_v3(t_min, t_max, t, data, T, print_report=True):
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]

    # =============================
    # STEP 1: 构造 jackknife samples
    # =============================
    jk_means = []
    for i in range(N_cfg):
        jk_indices = np.concatenate([np.arange(i), np.arange(i+1, N_cfg)])
        jk_data_subset = data[jk_indices][:, fit_mask]
        jk_mean = np.mean(jk_data_subset, axis=0)
        jk_means.append(jk_mean)
    jk_means = np.array(jk_means)  # shape (N_cfg, n_tfit)

    # =============================
    # STEP 2: 计算全局误差（固定权重）
    # =============================
    jk_mean_overall = np.mean(jk_means, axis=0)
    diffs = jk_means - jk_mean_overall
    # jackknife 方差公式 (N-1)/N * sum (diff^2)
    sigma_global = np.sqrt((N_cfg - 1) / N_cfg * np.sum(diffs**2, axis=0))
    # 这里如果要做相关拟合，也可以在这一步构造协方差矩阵 C

    # =============================
    # STEP 3: 对每个 jackknife sample 做拟合
    # 使用固定的 sigma_global 作为误差
    # =============================
    jk_params_A0 = []
    jk_params_m0 = []
    failed_fits = 0

    for i in range(N_cfg):
        jk_data_fit = jk_means[i]

        params = Parameters()
        params.add('A0', value=jk_data_fit[0], min=0)
        params.add('m0', value=0.5, min=0)

        try:
            result = minimize(
                residual, params, method='least_squares',
                kws={
                    "t_fit": t_fit,
                    "data_fit": jk_data_fit,
                    "err_fit": sigma_global,  # ✅固定的sigma
                    "T": T
                }
            )
            if result.success:
                jk_params_A0.append(result.params['A0'].value)
                jk_params_m0.append(result.params['m0'].value)
            else:
                failed_fits += 1

        except Exception:
            failed_fits += 1

    if len(jk_params_A0) == 0:
        raise RuntimeError("所有jackknife拟合都失败了")

    # =============================
    # STEP 4: 计算参数的 jackknife 误差
    # =============================
    jk_params_A0 = np.array(jk_params_A0)
    jk_params_m0 = np.array(jk_params_m0)

    A0_mean = np.mean(jk_params_A0)
    A0_std = np.sqrt((N_cfg - 1) * np.var(jk_params_A0, ddof=0))

    m0_mean = np.mean(jk_params_m0)
    m0_std = np.sqrt((N_cfg - 1) * np.var(jk_params_m0, ddof=0))

    # =============================
    # STEP 5: 输出
    # =============================
    class JackknifeResult:
        def __init__(self, A0_mean, A0_std, m0_mean, m0_std, success_rate):
            self.params = {
                'A0': type('', (), {'value': A0_mean, 'stderr': A0_std})(),
                'm0': type('', (), {'value': m0_mean, 'stderr': m0_std})()
            }
            self.success = True
            self.chisqr = None
            self.success_rate = success_rate

    if print_report:
        print("=" * 50)
        print("Jackknife拟合结果:")
        print(f"成功率: {len(jk_params_A0)}/{N_cfg} ({100*len(jk_params_A0)/N_cfg:.1f}%)")
        print(f"A0: {A0_mean:.6f} ± {A0_std:.6f}")
        print(f"m0: {m0_mean:.6f} ± {m0_std:.6f}")

    return JackknifeResult(
        A0_mean, A0_std,
        m0_mean, m0_std,
        len(jk_params_A0) / N_cfg
    )



def main(tmin,tmax):
    data, t, N_cfg = data_load()
    mean_corr = np.mean(data, axis=0)
    T = 96

    result= jackknife_fit_v2(t_min=tmin, t_max=tmax, t=t, data=data,T=T,print_report=True)

    return result


def test(data):
    N = data.shape[0]
    def func(data):
        return np.mean(data,axis=0)
    
    jk_sample,*_= jackknife(f=func,data=data,numb_blocks=N,conf_axis=0,return_sample=True)
    jk_means = np.mean(jk_sample,axis=1)

    print(f"mean shape:{jk_means.shape}")
    # print(f"err shape:{jk_err.shape}")
    print(f"sample shape:{jk_sample.shape}")


if __name__ == "__main__":

    out = main(5,48)
    # data,t,N = data_load()
    # test(data=data)