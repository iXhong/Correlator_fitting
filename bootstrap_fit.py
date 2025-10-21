import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report
import glob


def data_load():
    """ 
    load correlator data from .dat files
    Return:
        flip_ fliped & averaged data
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

    return flip_data, t, N


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


import numpy as np
from lmfit import minimize, Parameters


def bootstrap_fit(t_min: int,t_max: int,t,data,T: int,n_resamples: int = 500,print_report: bool = True,random_seed: int = 1234):
    """
    uncorrelated 1 state bootstrap fit
    Args:
        t_min, t_max: 拟合区间
        t: 时间点数组
        data: shape (N_cfg, N_t)
        T: Euclidean 时间长度
        n_resamples: bootstrap 重抽样次数
        print_report: 是否打印结果
    Returns:
        BootstrapResult: 结构与 JackknifeResult 相同
    """

    rng = np.random.default_rng(random_seed)
    N_cfg = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]

    # 构造 bootstrap samples (对配置取平均)
    bs_means = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, N_cfg, N_cfg)  # 有放回重抽样
        bs_sample = data[chosen][:, fit_mask]
        bs_mean = np.mean(bs_sample, axis=0)
        bs_means.append(bs_mean)
    bs_means = np.array(bs_means)  # shape (n_resamples, n_tfit)

    # 全局误差（每个时间点的标准差）
    sigma_global = np.std(bs_means, axis=0, ddof=1)

    # 对每个 bootstrap sample 拟合
    bs_params_A0 = []
    bs_params_m0 = []
    bs_redchi = []
    bs_aicc = []
    failed_fits = 0

    for i in range(n_resamples):
        bs_data_fit = bs_means[i]

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
                    "err_fit": sigma_global,
                    "T": T,
                },
            )
            if result.success:
                bs_params_A0.append(result.params["A0"].value)
                bs_params_m0.append(result.params["m0"].value)
                bs_redchi.append(result.redchi)
                bs_aicc.append(result.aic)
            else:
                failed_fits += 1
        except Exception:
            failed_fits += 1

    if len(bs_params_A0) == 0:
        raise RuntimeError("所有 bootstrap 拟合都失败了")

    # Bootstrap 误差：直接用标准差
    bs_params_A0 = np.array(bs_params_A0)
    bs_params_m0 = np.array(bs_params_m0)

    A0_mean = np.mean(bs_params_A0)
    A0_std = np.std(bs_params_A0, ddof=1)

    m0_mean = np.mean(bs_params_m0)
    m0_std = np.std(bs_params_m0, ddof=1)

    redchi = np.mean(bs_redchi)
    aicc = np.mean(bs_aicc)

    # 输出结构类
    class BootstrapResult:
        def __init__(self, A0_mean, A0_std, m0_mean, m0_std, redchi, aicc, success_rate):
            self.params = {
                "A0": type("", (), {"value": A0_mean, "stderr": A0_std})(),
                "m0": type("", (), {"value": m0_mean, "stderr": m0_std})(),
            }
            self.success = True
            self.redchi = redchi
            self.aicc = aicc
            self.success_rate = success_rate

    if print_report:
        print("=" * 50)
        print("Bootstrap拟合结果:")
        print(
            f"成功率: {len(bs_params_A0)}/{n_resamples} ({100 * len(bs_params_A0) / n_resamples:.1f}%)"
        )
        print(f"A0: {A0_mean} ± {A0_std}")
        print(f"m0: {m0_mean:.6f} ± {m0_std:.6f}")

    return BootstrapResult(
        A0_mean,
        A0_std,
        m0_mean,
        m0_std,
        redchi,
        aicc,
        len(bs_params_A0) / n_resamples,
    )


def plot_result(result, mean_corr, t, T):
    A0 = result.params["A0"].value
    m0 = result.params["m0"].value

    y = A0 * np.cosh(m0 * (t - T / 2))
    # print(y[5:16])
    # print(mean_corr[5:16])

    plt.figure()
    plt.scatter(t, np.log(y))
    plt.scatter(t, np.log(np.abs(mean_corr)))
    plt.show()


def main(tmin, tmax, show_plot, print_report, n_resamples=50):
    data, t, _ = data_load()
    mean_corr = np.mean(data, axis=0)
    T = 96

    # result, t_fit, data_fit, err_fit = bootstrap_fit(t_min=tmin, t_max=tmax, t=t, data=data, T=T, n_bootstrap=n_bootstrap, print_report=print_report)
    # result = bootstrap_fit(t_min=tmin, t_max=tmax, t=t, data=data, T=T, n_bootstrap=n_bootstrap, print_report=print_report)

    result = bootstrap_fit(t_min=tmin,t_max=tmax,t=t, data=data,T=T,n_resamples=n_resamples,print_report=print_report)

    if show_plot:
        plot_result(result, mean_corr, t, T)

    return result


if __name__ == "__main__":
    out = main(tmin=5, tmax=35, show_plot=False, print_report=True, n_resamples=10)
    

    # print(out.redchi)
    # print(out.aicc)