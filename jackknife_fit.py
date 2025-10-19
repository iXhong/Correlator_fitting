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


def jackknife_fit(t_min: int, t_max: int, t, data, T: int, print_report: bool = True):
    """
    uncorrelated 1 state jackknife fit
    Args:
        t_min
        t_max
        print_report:print report or not
    Returns:
        JackknifeResult: a class like params in lmfit
    Examples:
        m0 = result.params['m0'].value
    """

    N_blocks = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]

    # 构造 jackknife samples
    jk_means = []
    for i in range(N_blocks):
        jk_indices = np.concatenate([np.arange(i), np.arange(i + 1, N_blocks)])
        jk_data_subset = data[jk_indices][:, fit_mask]
        jk_mean = np.mean(jk_data_subset, axis=0)
        jk_means.append(jk_mean)
    jk_means = np.array(jk_means)  # shape (N_blocks, n_tfit)

    # 计算全局误差
    jk_mean_overall = np.mean(jk_means, axis=0)
    diffs = jk_means - jk_mean_overall
    # jackknife 方差公式 (N-1)/N * sum (diff^2)
    sigma_global = np.sqrt((N_blocks - 1) / N_blocks * np.sum(diffs**2, axis=0))
    # 这里如果要做相关拟合，也可以在这一步构造协方差矩阵 C

    # 对每个 jackknife sample 做拟合
    jk_params_A0 = []
    jk_params_m0 = []
    jk_redchi = []
    jk_aicc = []
    failed_fits = 0

    for i in range(N_blocks):
        jk_data_fit = jk_means[i]

        params = Parameters()
        params.add("A0", value=jk_data_fit[0], min=0)
        params.add("m0", value=0.5, min=0)

        try:
            result = minimize(
                residual,
                params,
                method="least_squares",
                kws={
                    "t_fit": t_fit,
                    "data_fit": jk_data_fit,
                    "err_fit": sigma_global,
                    "T": T,
                },
            )
            if result.success:
                jk_params_A0.append(result.params["A0"].value)
                jk_params_m0.append(result.params["m0"].value)
                jk_redchi.append(result.redchi)
                jk_aicc.append(result.aic)
            else:
                failed_fits += 1

        except Exception:
            failed_fits += 1

    if len(jk_params_A0) == 0:
        raise RuntimeError("所有jackknife拟合都失败了")

    # 计算参数的 jackknife 误差
    # p0_std = sqrt(N-1)*std(p0)
    jk_params_A0 = np.array(jk_params_A0)
    jk_params_m0 = np.array(jk_params_m0)

    A0_mean = np.mean(jk_params_A0)
    A0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_A0, ddof=0))

    m0_mean = np.mean(jk_params_m0)
    m0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_m0, ddof=0))

    redchi = np.mean(jk_redchi)
    aicc = np.mean(jk_aicc)

    # 输出
    class JackknifeResult:
        def __init__(self, A0_mean, A0_std, m0_mean, m0_std, redchi,aicc,success_rate):
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
        print("Jackknife拟合结果:")
        print(
            f"成功率: {len(jk_params_A0)}/{N_blocks} ({100 * len(jk_params_A0) / N_blocks:.1f}%)"
        )
        print(f"A0: {A0_mean} ± {A0_std}")
        print(f"m0: {m0_mean:.6f} ± {m0_std:.6f}")

    return JackknifeResult(
        A0_mean, A0_std, m0_mean, m0_std, redchi, aicc, len(jk_params_A0) / N_blocks
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


def main(tmin, tmax, show_plot, print_report):
    data, t,_ = data_load()
    mean_corr = np.mean(data, axis=0)
    T = 96

    # result, t_fit, data_fit, err_fit = jackknife_fit(t_min=tmin, t_max=tmax, t=t, data=data,T=T,print_report=print_report)
    result = jackknife_fit(
        t_min=tmin, t_max=tmax, t=t, data=data, T=T, print_report=print_report
    )

    if show_plot:
        # plot_fit_result(t, mean_corr, result, t_fit, data_fit)
        # plot_fit_result(t=t,mean_corr=mean_corr,jk_err=)
        plot_result(result, mean_corr, t, T)

    return result


if __name__ == "__main__":
    out = main(tmin=5, tmax=48, show_plot=False, print_report=True)

    # print(out.redchi)
    print(out.aicc)

    # print(f"m0:{out.params['m0'].value}")
    # print(f"A0:{out.params['A0'].value}")
    # print(out.params["A0"].stderr)
