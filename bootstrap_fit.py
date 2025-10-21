import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
import glob


def data_load():
    """ 
    load correlator data from .dat files
    Return:
        flip_data: fliped & averaged data
        t: time data
        N: number of correlator files
    """
   
    file_list = sorted(glob.glob("./mass/*.dat"))

    real_data_all = []

    for fname in file_list:
        data = np.loadtxt(fname, comments="#")
        filtered = data[data[:, 3] == 0]  # 第4列是 mumu
        real_values = filtered[:, 5]
        real_data_all.append(real_values)
    C_array = np.array(real_data_all)

    N = C_array.shape[0]  # files num
    flip_data = (C_array[:, :48] + np.flip(C_array[:, -48:])) / 2  # flip & average data
    t = np.arange(flip_data.shape[1])
    print(f"{N} 组数据，{len(t)} 个时间点")

    return flip_data, t,N


def fit_function_cosh(params, t, T):
    A0 = params["A0"]
    m0 = params["m0"]
    return A0 * np.cosh(m0 * (t - T / 2))


def residual(params, t_fit, data_fit, err_fit, T):
    """
    residual function to be minimized in Weighted LeastSquare method 
    Args:
    Return:
    """
    model = fit_function_cosh(params, t_fit, T)
    return (data_fit - model) / err_fit  


def bootstrap_fit(t_min: int, t_max: int, t, data, T: int, resample_times: int, print_report: bool = True):
    """
    Bootstrap重采样拟合
    Args:
        t_min: 最小时间
        t_max: 最大时间 
        t: 时间数组
        data: 数据矩阵 (N_samples, T_times)
        T: 时间步数
        resample_times: bootstrap重采样次数
        print_report: 是否打印报告
    Returns:
        dict: 包含均值、标准差和拟合范围的结果
    """
    DEFAULTSEED = 42
    N_blocks, T = data.shape
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    
    # Bootstrap重采样
    bs_samples = np.zeros((resample_times, T))
    
    for i in range(resample_times):
        rng = np.random.default_rng(DEFAULTSEED + i)
        chosen = rng.integers(0, N_blocks, N_blocks)
        bs_samples[i] = np.mean(data[chosen, :], axis=0)
    
    # 计算统计量
    mean = np.mean(bs_samples, axis=0)
    variance = np.sum((bs_samples - mean) ** 2, axis=0) / (resample_times - 1)
    sigma = np.sqrt(variance)


    # 这里如果要做相关拟合，也可以在这一步构造协方差矩阵 C

    # 对每个 bootstrap sample 做拟合
    bs_A0 = []
    bs_m0 = []
    failed_fits = 0

    for i in range(resample_times):
        bs_data_fit = bs_samples[i,fit_mask]
        sigma_fit = sigma[fit_mask]

        params = Parameters()
        params.add("A0", value=bs_data_fit[0], min=0)
        params.add("m0", value=0.5, min=0)

        try:
            result = minimize(
                residual,
                params,
                method="least_squares",
                kws={
                    "t_fit": t_fit,
                    "data_fit": bs_data_fit,
                    "err_fit": sigma_fit,
                    "T": T,
                },
            )
            if result.success:
                bs_A0.append(result.params["A0"].value)
                bs_m0.append(result.params["m0"].value)
            else:
                failed_fits += 1

        except Exception:
            failed_fits += 1

    if len(bs_A0) == 0:
        raise RuntimeError("所有bootstrap拟合都失败了")

    # 计算参数的 jackknife 误差
    # p0_std = sqrt(N-1)*std(p0)
    bs_A0 = np.array(bs_A0)
    bs_m0 = np.array(bs_m0)

    A0_mean = np.mean(bs_A0)
    # A0_std = np.sqrt((N_blocks - 1) * np.var(bs_A0, ddof=0))
    A0_std = np.std(bs_A0,ddof=1)

    m0_mean = np.mean(bs_m0)
    # m0_std = np.sqrt((N_blocks - 1) * np.var(bs_m0, ddof=0))
    m0_std = np.std(bs_m0, ddof=1)

    # 输出
    class BootstrapResult:
        def __init__(self, A0_mean, A0_std, m0_mean, m0_std, success_rate):
            self.params = {
                "A0": type("", (), {"value": A0_mean, "stderr": A0_std})(),
                "m0": type("", (), {"value": m0_mean, "stderr": m0_std})(),
            }
            self.success = True
            self.chisqr = None
            self.success_rate = success_rate

    if print_report:
        print("=" * 50)
        print("Bootstrap拟合结果:")
        print(
            f"成功率: {len(bs_A0)}/{resample_times} ({100 * len(bs_A0) / resample_times:.1f}%)"
        )
        print(f"A0: {A0_mean} ± {A0_std}")
        print(f"m0: {m0_mean:.6f} ± {m0_std:.6f}")

    return BootstrapResult(
        A0_mean, A0_std, m0_mean, m0_std, len(bs_A0) / resample_times
    )



def main(tmin,tmax,print_report,resample_times=42):
    data,t,_ = data_load()
    T = 96

    result = bootstrap_fit(
        t_min=tmin, t_max=tmax, t=t, data=data, T=T, print_report=print_report,resample_times=resample_times
    )
    return result



if __name__ == "__main__":
    out = main(tmin=5, tmax=48, print_report=True,resample_times=100)