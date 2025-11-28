"""
Description: bootstrap fit for one state and two state cosh functions
@author: George Liu
@since: 2025.11.19
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import least_squares

from plot import plot_histogram


def one_cosh_func(params, t, T):
    # params is ndarray: [A0, m0]
    A0, m0 = params
    return A0 * np.cosh(m0 * (t - T / 2))


def two_cosh_func(params, t, T):
    # params is ndarray: [A0, m0, A1, m1]
    A0, m0, A1, m1 = params
    return A0 * np.cosh(m0 * (t - T / 2)) + A1 * np.cosh(m1 * (t - T / 2))


def residuals_one_cosh(params, t_data, y_data, y_err, T):
    model = one_cosh_func(params, t_data, T)
    return (y_data - model) / y_err


def residuals_two_cosh(params, t_data, y_data, y_err, T):
    model = two_cosh_func(params, t_data, T)
    return (y_data - model) / y_err


def one_state_direct_fit(t_fit, y_fit, y_err_fit, T, initial_params, bounds):

    result = least_squares(
        residuals_one_cosh,
        x0=np.array([initial_params["A0"], initial_params["m0"]]),
        args=(t_fit, y_fit, y_err_fit, T),
        bounds=bounds,
        method="trf",
    )

    fitted_params = {"A0": result.x[0], "m0": result.x[1]}
    redchi2 = get_redchi2(result, len(y_fit), len(initial_params))
    chi2 = get_chi2(result)
    # redchi2 = get_redchi2_manual(fitted_params, t_fit, y_fit, y_err_fit, T)
    return fitted_params, result.success, chi2, redchi2


def two_state_direct_fit(t_fit, y_fit, y_err_fit, T, initial_params, bounds):

    result = least_squares(
        residuals_two_cosh,
        x0=np.array(
            [
                initial_params["A0"],
                initial_params["m0"],
                initial_params["A1"],
                initial_params["m1"],
            ]
        ),
        args=(t_fit, y_fit, y_err_fit, T),
        bounds=bounds,
        method="trf",
    )

    fitted_params = {
        "A0": result.x[0],
        "m0": result.x[1],
        "A1": result.x[2],
        "m1": result.x[3],
    }

    redchi2 = get_redchi2(result, len(y_fit), len(initial_params))
    chi2 = get_chi2(result)
    return fitted_params, result.success, chi2, redchi2


def one_state_bootstrap_fit(
    t_fit,
    bs_samples,
    sigma_fit,
    T,
    initial_params,
    bounds,
):
    n_bootstrap = bs_samples.shape[0]
    fitted_params_list = []
    success_list = []
    redchi2_list = []
    chi2_list = []

    for i in range(n_bootstrap):
        y_fit = bs_samples[i]

        fitted_params, success, chi2, redchi2 = one_state_direct_fit(
            t_fit, y_fit, sigma_fit, T, initial_params, bounds
        )
        fitted_params_list.append(fitted_params)
        success_list.append(success)
        chi2_list.append(chi2)
        redchi2_list.append(redchi2)

    return fitted_params_list, success_list, chi2_list, redchi2_list


def two_state_bootstrap_fit(
    t_fit,
    bs_samples,
    sigma_fit,
    T,
    initial_params,
    bounds,
):
    n_bootstrap = bs_samples.shape[0]
    fitted_params_list = []
    success_list = []
    chi2_list = []
    redchi2_list = []

    for i in range(n_bootstrap):
        y_fit = bs_samples[i]

        fitted_params, success, chi2, redchi2 = two_state_direct_fit(
            t_fit, y_fit, sigma_fit, T, initial_params, bounds
        )
        fitted_params_list.append(fitted_params)
        success_list.append(success)
        chi2_list.append(chi2)
        redchi2_list.append(redchi2)

    return fitted_params_list, success_list, chi2_list, redchi2_list


def prepare_p2_bs_samples(path, isCorrelated=False):
    # sigma used for uncorrelated fitting
    bs_samples = np.load(path)  # shape (n_bootstrap, n_data)
    # sigma = np.std(bs_samples, axis=0, ddof=1) / (len(bs_samples))
    sigma = np.std(bs_samples, axis=0, ddof=1)
    # cov_matrix = np.cov(bs_samples, rowvar=False)
    # if isCorrelated:
    #     sigma = cov_matrix
    # else:
    #     sigma =

    return bs_samples, sigma


def run_one_state_fit(path, T, tmin, tmax, savepath=None):
    bs_samples, sigma = prepare_p2_bs_samples(path)
    t = np.arange(len(sigma))
    mask = (t >= tmin) & (t <= tmax)
    t_fit = t[mask]
    bs_samples_fit = bs_samples[:, mask]
    sigma_fit = sigma[mask]

    initial_params = {"A0": 1e-15, "m0": 0.6}
    bounds = ([0, 0.55], [1e-14, 0.70])
    fitted_params_list, success_list, chi2_list, redchi2_list = one_state_bootstrap_fit(
        t_fit, bs_samples_fit, sigma_fit, T, initial_params, bounds
    )

    A0_list = [params["A0"] for params in fitted_params_list if params is not None]
    m0_list = [params["m0"] for params in fitted_params_list if params is not None]
    print("Number of successful fits:", sum(success_list), "out of", len(success_list))
    result_dict = {
        "A0_list": A0_list,
        "m0_list": m0_list,
        "success_list": success_list,
        "chi2_list": chi2_list,
        "redchi2_list": redchi2_list,
    }
    if savepath is not None:
        np.savez(savepath, **result_dict)
        print(f"Saved fit results to {savepath}")
    return result_dict


def run_two_state_fit(path, T, tmin, tmax, savepath=None):
    bs_samples, sigma = prepare_p2_bs_samples(path)
    t = np.arange(len(sigma))
    mask = (t >= tmin) & (t <= tmax)
    t_fit = t[mask]
    bs_samples_fit = bs_samples[:, mask]
    sigma_fit = sigma[mask]

    initial_params = {"A0": 3.65e-15, "m0": 0.5961, "A1": 2.84e-25, "m1": 1.10}
    bounds = ([0, 0.55, 0, 0.8], [1e-14, 0.65, 1e-24, 1.3])

    fitted_params_list, success_list, chi2_list, redchi2_list = two_state_bootstrap_fit(
        t_fit, bs_samples_fit, sigma_fit, T, initial_params, bounds
    )
    A0_list = [params["A0"] for params in fitted_params_list if params is not None]
    m0_list = [params["m0"] for params in fitted_params_list if params is not None]
    A1_list = [params["A1"] for params in fitted_params_list if params is not None]
    m1_list = [params["m1"] for params in fitted_params_list if params is not None]
    print("Number of successful fits:", sum(success_list), "out of", len(success_list))
    result_dict = {
        "A0_list": A0_list,
        "m0_list": m0_list,
        "A1_list": A1_list,
        "m1_list": m1_list,
        "success_list": success_list,
        "chi2_list": chi2_list,
        "redchi2_list": redchi2_list,
    }
    if savepath is not None:
        np.savez(savepath, **result_dict)
        print(f"Saved fit results to {savepath}")
    return result_dict


def result_plot(t, y_mean, y_err, result_dict, T, state="one"):
    print(t.shape, y_mean.shape, y_err.shape)
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        t,
        y_mean,
        y_err,
        fmt="o",
        label=r"$data\ \hat{p}^2=0$",
        markersize=4,
    )

    plt.plot(t, y_fit, label=label_fit, color="red")
    plt.xlabel("t")
    plt.ylabel("C(t)")
    plt.yscale("log")
    plt.title("Bootstrap Fit Result")
    plt.legend()
    plt.grid()
    plt.show()


def get_redchi2(result, n_data, n_param):
    dof = n_data - n_param
    redchi2 = np.sum(result.fun**2) / dof
    return redchi2


def get_chi2(result):
    chi2 = np.sum(result.fun**2)
    return chi2


def scan_tmin_two_state(
    path,
    T,
    tmin_start=1,
    tmin_end=24,
    tmax=30,
    outdir="./data/processed/mom/bs_fit_results/scan_two_state",
):
    """
    扫描 tmin (固定 tmax=30) 进行 two-state bootstrap fit。

    对每个 tmin 产生文件：
       two_state_tmin_{tmin}_tmax_{tmax}.npz
    """

    os.makedirs(outdir, exist_ok=True)

    print(f"[Two-state Scan] tmin={tmin_start} → {tmin_end} (tmax={tmax})")
    print("-" * 60)

    for tmin in range(tmin_start, tmin_end + 1):

        savefile = f"{outdir}/two_state_tmin_{tmin}_tmax_{tmax}.npz"

        print(f"Running two-state fit: tmin={tmin}, tmax={tmax}")

        result_dict = run_two_state_fit(
            path=path,
            T=T,
            tmin=tmin,
            tmax=tmax,
            savepath=savefile,
        )

        avg_redchi2 = np.mean(result_dict["redchi2_list"])
        print(f"Saved: {savefile}, avg redchi2 = {avg_redchi2:.5f}")

    print("Two-state scan finished.\n")


def scan_tmin_one_state(
    path,
    T,
    tmin_start=1,
    tmin_end=24,
    tmax=30,
    outdir="./data/processed/mom/bs_fit_results/scan_one_state",
):
    """
    扫描 tmin (固定 tmax=30) 进行 one-state bootstrap fit。

    对每个 tmin 产生文件：
       one_state_tmin_{tmin}_tmax_{tmax}.npz
    """

    os.makedirs(outdir, exist_ok=True)

    print(f"[One-state Scan] tmin={tmin_start} → {tmin_end} (tmax={tmax})")
    print("-" * 60)

    for tmin in range(tmin_start, tmin_end + 1):

        savefile = f"{outdir}/one_state_tmin_{tmin}_tmax_{tmax}.npz"

        print(f"Running one-state fit: tmin={tmin}, tmax={tmax}")

        result_dict = run_one_state_fit(
            path=path,
            T=T,
            tmin=tmin,
            tmax=tmax,
            savepath=savefile,
        )

        avg_redchi2 = np.mean(result_dict["redchi2_list"])
        print(f"Saved: {savefile}, avg redchi2 = {avg_redchi2:.5f}")

    print("One-state scan finished.\n")


def get_redchi2_nmin(path):
    filelist = sorted(glob.glob(os.path.join(path, "two_state_tmin_*_tmax_*.npz")))
    tmin_list = []
    redchi2_list = []
    for filepath in filelist:
        data = np.load(filepath)
        tmin_str = os.path.basename(filepath).split("_")[3]
        tmin = int(tmin_str)
        tmin_list.append(tmin)

        redchi2_vals = data["redchi2_list"]
        avg_redchi2 = np.mean(redchi2_vals)
        redchi2_list.append(avg_redchi2)
    return tmin_list, redchi2_list


def add_chi2(path, n_param=4):
    """
    根据保存的 redchi2_list 计算并补充 chi2_list 。

    参数
    ----
    path : str
        包含 *.npz 拟合结果文件的目录。
        文件名假定形如:
            one_state_tmin_{tmin}_tmax_{tmax}.npz
            two_state_tmin_{tmin}_tmax_{tmax}.npz
    n_param : int
        拟合参数个数，默认 4（two-state）。对于 one-state 可显式传入 2。
    """
    filelist = sorted(glob.glob(os.path.join(path, "*_tmin_*_tmax_*.npz")))
    for filepath in filelist:
        data = np.load(filepath)
        redchi2_vals = data["redchi2_list"]

        basename = os.path.basename(filepath).replace(".npz", "")
        parts = basename.split("_")

        # 解析 tmin / tmax
        try:
            i_tmin = parts.index("tmin")
            tmin = int(parts[i_tmin + 1])
            i_tmax = parts.index("tmax")
            tmax = int(parts[i_tmax + 1])
        except ValueError:
            # 回退到旧的命名规则: {state}_tmin_{tmin}_tmax_{tmax}
            tmin = int(parts[3])
            tmax = int(parts[5])

        n_data = tmax - tmin + 1
        dof = n_data - n_param

        chi2_vals = [redchi2 * dof for redchi2 in redchi2_vals]

        save_dict = {key: data[key] for key in data.files}
        save_dict["chi2_list"] = chi2_vals

        np.savez(filepath, **save_dict)
        print(f"Updated: {os.path.basename(filepath)}")
    print("all update")


def get_aicc(chi2, n_data, n_param):
    """
    Calculate the corrected Akaike Information Criterion (AICc).

    在高斯误差、以标准化残差定义 chi2 的情形下，常用的形式是
        AIC  = chi2 + 2*k
        AICc = chi2 + 2*k + 2*k*(k+1)/(n - k - 1)

    其中 k = n_param 为参数个数，n = n_data 为参与拟合的数据点数。
    这里忽略与模型无关的常数项。
    """
    aic = chi2 + 2 * n_param
    correction = (2 * n_param * (n_param + 1)) / (n_data - n_param - 1)
    return aic + correction


def get_bootstrap_aicc(path):
    """
    读取指定目录下的 *_tmin_*_tmax_*.npz 文件，利用其中的 chi2_list
    计算各个 tmin 的平均 AICc，返回 {tmin: <AICc 平均>} 的字典。

    要求 npz 文件名形如:
        one_state_tmin_{tmin}_tmax_{tmax}.npz   (k=2)
        two_state_tmin_{tmin}_tmax_{tmax}.npz   (k=4)
    并且文件内已包含 chi2_list。
    """
    filelist = sorted(glob.glob(os.path.join(path, "*_tmin_*_tmax_*.npz")))
    aicc_results = {}

    for filepath in filelist:
        data = np.load(filepath)
        chi2_vals = data["chi2_list"]

        basename = os.path.basename(filepath).replace(".npz", "")
        parts = basename.split("_")

        # 判定是一态还是二态
        if basename.startswith("one_state"):
            n_param = 2
        elif basename.startswith("two_state"):
            n_param = 4
        else:
            raise ValueError(
                f"Cannot infer number of parameters from filename: {basename}"
            )

        # 解析 tmin / tmax
        try:
            i_tmin = parts.index("tmin")
            tmin = int(parts[i_tmin + 1])
            i_tmax = parts.index("tmax")
            tmax = int(parts[i_tmax + 1])
        except ValueError:
            # 回退到旧的命名规则: {state}_tmin_{tmin}_tmax_{tmax}
            tmin = int(parts[3])
            tmax = int(parts[5])

        n_data = tmax - tmin + 1

        aicc_vals = [get_aicc(chi2, n_data, n_param) for chi2 in chi2_vals]
        avg_aicc = np.mean(aicc_vals)

        aicc_results[tmin] = avg_aicc

    return aicc_results


def get_bootstrap_m0(path):
    """
    读取指定目录下的 *_tmin_*_tmax_*.npz 文件，返回 {tmin: (m0_mean,m0_error)} 的字典。

    要求 npz 文件名形如:
        one_state_tmin_{tmin}_tmax_{tmax}.npz
        two_state_tmin_{tmin}_tmax_{tmax}.npz
    并且文件内已包含 m0_list。
    """
    filelist = sorted(glob.glob(os.path.join(path, "*_tmin_*_tmax_*.npz")))
    m0_results = {}

    for filepath in filelist:
        data = np.load(filepath)
        m0_vals = data["m0_list"]

        basename = os.path.basename(filepath).replace(".npz", "")
        parts = basename.split("_")

        # 解析 tmin / tmax
        try:
            i_tmin = parts.index("tmin")
            tmin = int(parts[i_tmin + 1])
        except ValueError:
            # 回退到旧的命名规则: {state}_tmin_{tmin}_tmax_{tmax}
            tmin = int(parts[3])

        m0_mean = np.mean(m0_vals)
        m0_error = np.std(m0_vals, ddof=1)
        m0_results[tmin] = (m0_mean, m0_error)

    return m0_results


def select_best_fit_by_aicc(one_state_dir, two_state_dir):
    """
    读取一态与二态目录中所有 tmin 对应的 AICc，
    对每个 tmin 比较两种模型的 AICc，
    返回 {tmin: (best_model, best_aicc)} 的字典。

    best_model ∈ {"one_state", "two_state"}
    """

    # 复用你已有的 AICc 读取函数
    aicc_one = get_bootstrap_aicc(one_state_dir)
    aicc_two = get_bootstrap_aicc(two_state_dir)

    # 收集所有 tmin（两个目录的并集）
    all_tmins = sorted(set(aicc_one.keys()) | set(aicc_two.keys()))

    best_results = {}

    for t in all_tmins:
        a1 = aicc_one.get(t, np.inf)
        a2 = aicc_two.get(t, np.inf)

        if a1 < a2:
            best_results[t] = ("one_state", a1)
        else:
            best_results[t] = ("two_state", a2)

    return best_results


def get_best_m0_sequence(best_selection, m0_one, m0_two):
    """
    根据 best_selection（AICc best）选择最优模型的 m0(tmin)
    输入:
        best_selection: {tmin: ("one_state"/"two_state", aicc)}
        m0_one: {tmin: (m_mean, m_err)}
        m0_two: {tmin: (m_mean, m_err)}
    输出:
        tmin_vals: sorted list
        m_best:    [m_mean_i]
        m_err:     [m_err_i]
    """
    tmin_vals = sorted(best_selection.keys())

    m_best = []
    m_err = []

    for t in tmin_vals:
        model, _ = best_selection[t]
        if model == "one_state":
            m, e = m0_one[t]
        else:
            m, e = m0_two[t]
        m_best.append(m)
        m_err.append(e)

    return tmin_vals, np.array(m_best), np.array(m_err)


def weighted_plateau_average(n_lower, n_upper, m_vals, m_errs):
    """
    加权 plateau average。

    参数：
        n_lower:     plateau 左端点 index
        n_upper:     plateau 右端点 index
        m_vals:      array of mean m0(tmin)
        m_errs:      array of error m0_err(tmin)
    返回：
        weighted_avg, weighted_stat_err, weighted_sys_err
    """
    m_plateau = m_vals[n_lower : n_upper + 1]
    err_plateau = m_errs[n_lower : n_upper + 1]

    partA = 1 / (err_plateau**2)
    averaged_m = np.sum(m_plateau * partA) / np.sum(partA)
    stat_err = np.sqrt(1 / np.sum(partA))
    sys_err = np.sqrt(np.mean((m_plateau - np.mean(m_plateau)) ** 2))

    return averaged_m, stat_err, sys_err


def plot_plateau(
    n_list,
    m_vals,
    m_errs,
    n_lower,
    n_upper,
    xlabel=r"$n_{\sigma,\min}$",
    ylabel=r"$M_{\text{scr2}}$",
    outpath=None,
):
    """
    画出类似论文中的 plateau 平均图。

    参数：
        n_list : 1D array, 横轴 (例如 n_sigma_min)
        m_vals : 1D array, 拟合得到的质量 M(tmin)
        m_errs : 1D array, 对应误差
        n_lower, n_upper : plateau 的左右端点（Python index）
        outpath : 若给出字符串，则保存到该路径；否则只显示
    """
    # 先做 plateau average
    m_avg, stat_err, sys_err = weighted_plateau_average(
        n_lower, n_upper, m_vals, m_errs
    )

    # 画图
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    # 所有点的误差条（红色）
    ax.errorbar(
        n_list,
        m_vals,
        yerr=m_errs,
        fmt="o",
        ms=3,
        color="red",
        ecolor="red",
        elinewidth=0.8,
        capsize=2,
        linestyle="none",
    )

    # 整个横轴范围（为了画带）
    x_min = np.min(n_list) - 0.5
    x_max = np.max(n_list) + 0.5

    # 红色：系统误差带
    ax.fill_between(
        [x_min, x_max],
        m_avg - sys_err,
        m_avg + sys_err,
        color="red",
        alpha=0.3,
        label="syst",
    )

    # 蓝色：统计误差带
    ax.fill_between(
        [x_min, x_max],
        m_avg - stat_err,
        m_avg + stat_err,
        color="blue",
        alpha=0.5,
        label="stat",
    )

    # 蓝色中心线（平均值）
    ax.axhline(m_avg, color="blue", linewidth=1.2)

    # （可选）用浅灰色标出 plateau 选取的区间
    ax.axvspan(
        n_list[n_lower] - 0.5,
        n_list[n_upper] + 0.5,
        color="grey",
        alpha=0.1,
        zorder=0,
    )

    # 标题 & 坐标轴
    ax.set_title("red: syst, blue: stat", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 细一点的坐标轴风格
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300)
    plt.show()

    # 顺便把数值也打印一下，方便检查
    print(f"plateau range: indices [{n_lower}, {n_upper}]")
    print(f"<M> = {m_avg:.6f}")
    print(f"stat err = {stat_err:.6f}")
    print(f"syst err = {sys_err:.6f}")


def gaussian_plateau_averaging(
    tmin_vals, m_vals, m_errs, plateau_range, n_samples=5000, plot=True
):
    """
    Gaussian plateau averaging following Sandmeyer PhD Sec. 4.3.4.

    参数：
        tmin_vals:   list of tmin
        m_vals:      array of mean m0(tmin)
        m_errs:      array of error m0_err(tmin)
        plateau_range: (tmin_left, tmin_right)，例如 (9,15)
        n_samples:   bootstrap 样本数
        plot:        是否绘图（论文风格）

    返回：
        median, (err_minus, err_plus), samples
    """

    # ---- 选取 plateau 区间 ----
    tmin_left, tmin_right = plateau_range

    mask = [(t >= tmin_left) and (t <= tmin_right) for t in tmin_vals]
    m_plateau = m_vals[mask]
    err_plateau = m_errs[mask]
    t_plateau = np.array(tmin_vals)[mask]

    # ---- Gaussian bootstrap ----
    # 对 plateau 中的每个点都生成 n_samples 个随机噪声
    # 假设它们 fully correlated → 合并为一个总体分布
    combined_samples = []

    for i in range(len(m_plateau)):
        # 每个点的 Gaussian 分布
        samples_i = np.random.normal(
            loc=m_plateau[i], scale=err_plateau[i], size=n_samples
        )
        combined_samples.append(samples_i)

    # shape = (num_plateau_points, n_samples)
    combined_samples = np.array(combined_samples)

    # 合并为一个一维分布（论文明确指出需要将所有点合并）
    # Flatten 所有 plateau 点的 sample
    merged = combined_samples.flatten()

    # ---- final statistics ----
    median = np.median(merged)
    low = np.percentile(merged, 16)
    high = np.percentile(merged, 84)
    err_minus = median - low
    err_plus = high - median

    # ---- 绘图（论文 Figure 4.7 风格） ----
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # (1) 左图：m(tmin) plateau 区域
        axes[0].errorbar(
            tmin_vals,
            m_vals,
            yerr=m_errs,
            fmt="o",
            color="gray",
            ecolor="gray",
            alpha=0.6,
            label="data",
        )
        axes[0].errorbar(
            t_plateau,
            m_plateau,
            yerr=err_plateau,
            fmt="o",
            color="black",
            ecolor="black",
            label="plateau",
        )

        axes[0].axhline(median, color="red", linestyle="--", label="median")
        axes[0].fill_between(
            tmin_vals, low, high, color="red", alpha=0.2, label="68% band"
        )

        axes[0].set_xlabel(r"$t_{\min}$")
        axes[0].set_ylabel(r"$m_0$")
        axes[0].set_title("Plateau region with Gaussian bootstrap")
        axes[0].grid(True)
        axes[0].legend()

        # (2) 右图：Histogram（论文 Figure 4.7 右图）
        axes[1].hist(merged, bins=50, color="gray", edgecolor="black", density=True)
        axes[1].axvline(median, color="red", linestyle="--", label="median")
        axes[1].axvline(low, color="blue", linestyle="--", label="16%")
        axes[1].axvline(high, color="blue", linestyle="--", label="84%")
        axes[1].set_title("Gaussian bootstrap distribution")
        axes[1].set_xlabel(r"$m_0$")
        axes[1].set_ylabel("Density")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return median, (err_minus, err_plus), merged


if __name__ == "__main__":
    # inpath = "./data/processed/mom/bs_samples/phi_p2_0_bs.npy"
    # outpath1 = "./data/processed/mom/bs_fit_results/one_state_fit_results1.npz"
    # outpath2 = "./data/processed/mom/bs_fit_results/two_state_fit_results2.npz"
    # outpath3 = "./data/processed/mom/bs_fit_results/two_state_fit_results3.npz"
    # inpath2 = "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_fit_results/scan_two_state/two_state_tmin_5_tmax_30.npz"
    savepath2 = "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_fit_results/test/two_state"
    savepath1 = "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_fit_results/test/one_state"

    best_fit = select_best_fit_by_aicc(savepath1, savepath2)
    m0_one = get_bootstrap_m0(savepath1)
    m0_two = get_bootstrap_m0(savepath2)
    # m0_one_means = [m0_one[t][0] for t in sorted(m0_one.keys())]
    # m0_two_means = [m0_two[t][0] for t in sorted(m0_two.keys())]
    # m0_one_errs = [m0_one[t][1] for t in sorted(m0_one.keys())]
    # m0_two_errs = [m0_two[t][1] for t in sorted(m0_two.keys())]
    tmin_vals, m0_best, m0_err = get_best_m0_sequence(best_fit, m0_one, m0_two)

    # plateau averaging
    # tmin = 6
    # tmax = 24

    # n_lower = tmin_vals.index(tmin)
    # n_upper = tmin_vals.index(tmax)
    # plot_plateau(
    #     n_list=tmin_vals,
    #     m_vals=m0_best,
    #     m_errs=m0_err,
    #     n_lower=n_lower,
    #     n_upper=n_upper,
    #     xlabel=r"$n_{\sigma,\min}$",
    #     ylabel=r"a$m_0$",
    #     outpath=None,
    # )

    # plot m0 of one state, two state and best selected
    # plt.figure(figsize=(8, 5))
    # plt.errorbar(
    #     tmin_vals,
    #     m0_one_means,
    #     yerr=m0_one_errs,
    #     fmt="o",
    #     capsize=3,
    #     label="one-state",
    # )
    # plt.errorbar(
    #     tmin_vals,
    #     m0_two_means,
    #     yerr=m0_two_errs,
    #     fmt="s",
    #     capsize=3,
    #     label="two-state",
    # )
    # plt.errorbar(
    #     tmin_vals,
    #     m0_best,
    #     yerr=m0_err,
    #     fmt="o",
    #     capsize=3,
    #     label="AICc selected",
    # )
    # plt.xlabel(r"$n_{\sigma,min}$")
    # plt.ylabel(r"a$m_0$")
    # plt.grid()
    # plt.legend()
    # plt.show()

    # compare aicc of one state and two state
    # plt.figure(figsize=(8, 5))
    # plt.plot(tmin_vals, aicc_vals1, marker="o")
    # plt.plot(tmin_vals, aicc_vals2, marker="s")
    # plt.plot(tmin_vals, aicc_min, marker="^")
    # plt.legend(["one-state", "two-state", "min(AICc)"])
    # plt.xlabel(r"$n_{\sigma,min}$")
    # plt.ylabel("AICc")
    # plt.ylim(0, 30)
    # plt.grid()
    # plt.show()

    # get m0 of diff tmin
    # m0 = get_bootstrap_m0(savepath2)
    # m0_one = get_bootstrap_m0(savepath1)
    # plt.figure(figsize=(8, 5))
    # tmin_vals = sorted(m0.keys())
    # m0_means = [m0[tmin][0] for tmin in tmin_vals]
    # m0_errors = [m0[tmin][1] for tmin in tmin_vals]
    # m0_one_means = [m0_one[tmin][0] for tmin in tmin_vals]
    # m0_one_errors = [m0_one[tmin][1] for tmin in tmin_vals]
    # plt.errorbar(
    #     tmin_vals,
    #     m0_means,
    #     yerr=m0_errors,
    #     fmt="o",
    #     capsize=3,
    # )
    # plt.errorbar(
    #     tmin_vals,
    #     m0_one_means,
    #     yerr=m0_one_errors,
    #     fmt="s",
    #     capsize=3,
    # )
    # plt.legend(["two-state", "one-state"])
    # plt.xlabel(r"$n_{\sigma,min}$")
    # plt.ylabel(r"a$m_0$")
    # plt.grid()
    # plt.show()

    # scan_tmin_one_state(
    #     path=inpath,
    #     T=96,
    #     tmin_start=1,
    #     tmin_end=24,
    #     tmax=30,
    #     outdir=savepath1,
    # )

    # m0 of diff tmin
    # tmin_list, m0_list = get_redchi2_nmin(savepath2)

    # # aicc of diff tmin
    # aicc_results1 = get_bootstrap_aicc(savepath1)
    # aicc_results2 = get_bootstrap_aicc(savepath2)
    # tmin_vals = sorted(aicc_results1.keys())
    # aicc_vals1 = [aicc_results1[tmin] for tmin in tmin_vals]
    # aicc_vals2 = [aicc_results2[tmin] for tmin in tmin_vals]

    # # print(aicc_vals1)
    # # print(aicc_vals2)

    # plt.figure(figsize=(8, 5))
    # plt.plot(tmin_vals, aicc_vals1, marker="o")
    # plt.plot(tmin_vals, aicc_vals2, marker="s")
    # plt.legend(["one-state", "two-state"])
    # plt.xlabel(r"$n_{\sigma,min}$")
    # plt.ylabel("AICc")
    # plt.ylim(0, 30)
    # plt.grid()
    # plt.show()
