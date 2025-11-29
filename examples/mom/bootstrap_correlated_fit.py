"""
bootstrap_correlated_fit.py

Description: bootstrap *correlated* fit for one-state and two-state cosh functions.
             Uses covariance matrix from bootstrap samples and Cholesky-based
             correlated residuals.

@author: George Liu
@since: 2025.11.29
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


# ----------------------------------------------------------------------
#  模型函数
# ----------------------------------------------------------------------
def one_cosh_func(params, t, T):
    """
    One-state cosh correlator:
        C(t) = A0 * cosh(m0 * (t - T/2))
    params: [A0, m0]
    """
    A0, m0 = params
    return A0 * np.cosh(m0 * (t - T / 2))


def two_cosh_func(params, t, T):
    """
    Two-state cosh correlator:
        C(t) = A0 * cosh(m0 * (t - T/2)) + A1 * cosh(m1 * (t - T/2))
    params: [A0, m0, A1, m1]
    """
    A0, m0, A1, m1 = params
    return A0 * np.cosh(m0 * (t - T / 2)) + A1 * np.cosh(m1 * (t - T / 2))


# ----------------------------------------------------------------------
#  协方差矩阵 & 相关残差构造
# ----------------------------------------------------------------------
def build_covariance(bs_samples_fit, mode="full", eps=1e-12):
    """
    由 bootstrap 样本构造协方差矩阵。

    参数
    ----
    bs_samples_fit : array, shape (N_bootstrap, N_data)
        限制到拟合区间 [tmin, tmax] 后的 bootstrap 样本。
    mode : {"full", "diag"}
        "full" : 使用完整协方差矩阵。
        "diag" : 将 off-diagonal 元素置 0，只保留对角线。
    eps : float
        对角线上加入的小 regularization，避免数值上非正定。

    返回
    ----
    cov_matrix : array, shape (N_data, N_data)
    """
    # np.cov 默认是无偏估计，rowvar=False 表示每一列是一个变量（对应一个时间点）
    cov = np.cov(bs_samples_fit, rowvar=False, ddof=1)

    if mode == "diag":
        cov = np.diag(np.diag(cov))

    # 数值 regularization：在对角线上加一份小的噪声，保证正定
    diag = np.diag(cov)
    avg_diag = np.mean(diag) if diag.size > 0 else 1.0
    jitter = eps * avg_diag if avg_diag > 0 else eps
    cov = cov + jitter * np.eye(cov.shape[0])

    return cov


def make_correlated_residuals(C, model_func, T):
    """
    构造 correlated residual 函数，使用 Cholesky 分解。

    给定协方差矩阵 C，我们做
        C = L L^T,   (L 下三角)
        C^{-1} = (L^{-1})^T L^{-1}

    对于残差 r = y - f(p, t)，
        chi^2 = r^T C^{-1} r
              = (L^{-1} r)^T (L^{-1} r)
    所以 least_squares 中的 residual 可以定义为
        res = L^{-1} r

    这里采用你给出的形式：
        def residuals(params, x, y):
            model = f(params, x)
            r = y - model
            return Linv @ r

    只是多加了 model_func 和 T 在闭包中。
    """
    L = np.linalg.cholesky(C)
    Linv = np.linalg.inv(L)

    def residuals(params, t_data, y_data):
        model = model_func(params, t_data, T)
        r = y_data - model
        return Linv @ r  # correlated residual

    return residuals


# ----------------------------------------------------------------------
#  单次 correlated 拟合（one-state, two-state）
# ----------------------------------------------------------------------
def get_redchi2(result, n_data, n_param):
    dof = n_data - n_param
    redchi2 = np.sum(result.fun**2) / dof
    return redchi2


def get_chi2(result):
    chi2 = np.sum(result.fun**2)
    return chi2


def one_state_direct_fit(
    t_fit,
    y_fit,
    cov_matrix,
    T,
    initial_params,
    bounds,
):
    """
    对单个 bootstrap 样本进行 one-state *correlated* 拟合。
    使用相同的 cov_matrix (由全部 bootstrap 样本给出)。
    """
    residuals = make_correlated_residuals(cov_matrix, one_cosh_func, T)

    result = least_squares(
        residuals,
        x0=np.array([initial_params["A0"], initial_params["m0"]]),
        args=(t_fit, y_fit),
        bounds=bounds,
        method="trf",
    )

    fitted_params = {"A0": result.x[0], "m0": result.x[1]}
    chi2 = get_chi2(result)
    redchi2 = get_redchi2(result, len(t_fit), len(initial_params))

    return fitted_params, result.success, chi2, redchi2


def two_state_direct_fit(
    t_fit,
    y_fit,
    cov_matrix,
    T,
    initial_params,
    bounds,
):
    """
    对单个 bootstrap 样本进行 two-state *correlated* 拟合。
    """
    residuals = make_correlated_residuals(cov_matrix, two_cosh_func, T)

    result = least_squares(
        residuals,
        x0=np.array(
            [
                initial_params["A0"],
                initial_params["m0"],
                initial_params["A1"],
                initial_params["m1"],
            ]
        ),
        args=(t_fit, y_fit),
        bounds=bounds,
        method="trf",
    )

    fitted_params = {
        "A0": result.x[0],
        "m0": result.x[1],
        "A1": result.x[2],
        "m1": result.x[3],
    }
    chi2 = get_chi2(result)
    redchi2 = get_redchi2(result, len(t_fit), len(initial_params))

    return fitted_params, result.success, chi2, redchi2


# ----------------------------------------------------------------------
#  bootstrap correlated 拟合（对所有 bootstrap 样本）
# ----------------------------------------------------------------------
def one_state_bootstrap_fit(
    t_fit,
    bs_samples_fit,
    cov_matrix,
    T,
    initial_params,
    bounds,
):
    """
    对所有 bootstrap 样本做 one-state correlated 拟合。
    """
    n_bootstrap = bs_samples_fit.shape[0]
    fitted_params_list = []
    success_list = []
    chi2_list = []
    redchi2_list = []

    for i in range(n_bootstrap):
        y_fit = bs_samples_fit[i]

        fitted_params, success, chi2, redchi2 = one_state_direct_fit(
            t_fit, y_fit, cov_matrix, T, initial_params, bounds
        )
        fitted_params_list.append(fitted_params)
        success_list.append(success)
        chi2_list.append(chi2)
        redchi2_list.append(redchi2)

    return fitted_params_list, success_list, chi2_list, redchi2_list


def two_state_bootstrap_fit(
    t_fit,
    bs_samples_fit,
    cov_matrix,
    T,
    initial_params,
    bounds,
):
    """
    对所有 bootstrap 样本做 two-state correlated 拟合。
    """
    n_bootstrap = bs_samples_fit.shape[0]
    fitted_params_list = []
    success_list = []
    chi2_list = []
    redchi2_list = []

    for i in range(n_bootstrap):
        y_fit = bs_samples_fit[i]

        fitted_params, success, chi2, redchi2 = two_state_direct_fit(
            t_fit, y_fit, cov_matrix, T, initial_params, bounds
        )
        fitted_params_list.append(fitted_params)
        success_list.append(success)
        chi2_list.append(chi2)
        redchi2_list.append(redchi2)

    return fitted_params_list, success_list, chi2_list, redchi2_list


# ----------------------------------------------------------------------
#  高层封装：读入文件、选取 [tmin, tmax]、构造协方差并做拟合
# ----------------------------------------------------------------------
def run_one_state_fit(
    path,
    T,
    tmin,
    tmax,
    savepath=None,
    cov_mode="full",
    eps=1e-12,
):
    """
    对给定 p^2、给定 [tmin, tmax] 区间做 one-state *correlated* bootstrap 拟合。

    参数
    ----
    path : str
        npy 文件路径，形状 (N_bootstrap, N_t)。
    T : int
        时间方向长度。
    tmin, tmax : int
        拟合时间窗口。
    savepath : str or None
        若给出，则将结果以 npz 保存。
    cov_mode : {"full", "diag"}
        "full": 使用完整协方差矩阵。
        "diag": 将 off-diagonal 置 0，但仍用 correlated 残差形式。
    eps : float
        协方差矩阵 regularization 参数。
    """
    bs_samples = np.load(path)  # shape (N_bootstrap, N_t)
    n_t = bs_samples.shape[1]
    t_all = np.arange(n_t)

    mask = (t_all >= tmin) & (t_all <= tmax)
    t_fit = t_all[mask]
    bs_samples_fit = bs_samples[:, mask]

    # 基于拟合窗口的 bootstrap 样本构造协方差矩阵
    cov_matrix = build_covariance(bs_samples_fit, mode=cov_mode, eps=eps)

    initial_params = {"A0": 1e-15, "m0": 0.6}
    bounds = ([0.0, 0.55], [1e-14, 0.70])

    fitted_params_list, success_list, chi2_list, redchi2_list = one_state_bootstrap_fit(
        t_fit, bs_samples_fit, cov_matrix, T, initial_params, bounds
    )

    A0_list = [params["A0"] for params in fitted_params_list if params is not None]
    m0_list = [params["m0"] for params in fitted_params_list if params is not None]

    print(
        "Number of successful one-state fits:",
        sum(success_list),
        "out of",
        len(success_list),
    )

    result_dict = {
        "A0_list": A0_list,
        "m0_list": m0_list,
        "success_list": success_list,
        "chi2_list": chi2_list,
        "redchi2_list": redchi2_list,
        "tmin": tmin,
        "tmax": tmax,
        "cov_mode": cov_mode,
    }

    if savepath is not None:
        np.savez(savepath, **result_dict)
        print(f"Saved correlated one-state fit results to {savepath}")

    return result_dict


def run_two_state_fit(
    path,
    T,
    tmin,
    tmax,
    savepath=None,
    cov_mode="full",
    eps=1e-12,
):
    """
    对给定 p^2、给定 [tmin, tmax] 区间做 two-state *correlated* bootstrap 拟合。
    """
    bs_samples = np.load(path)  # shape (N_bootstrap, N_t)
    n_t = bs_samples.shape[1]
    t_all = np.arange(n_t)

    mask = (t_all >= tmin) & (t_all <= tmax)
    t_fit = t_all[mask]
    bs_samples_fit = bs_samples[:, mask]

    cov_matrix = build_covariance(bs_samples_fit, mode=cov_mode, eps=eps)

    initial_params = {
        "A0": 3.65e-15,
        "m0": 0.5961,
        "A1": 2.84e-25,
        "m1": 1.10,
    }
    bounds = ([0.0, 0.55, 0.0, 0.8], [1e-14, 0.65, 1e-24, 1.3])

    fitted_params_list, success_list, chi2_list, redchi2_list = two_state_bootstrap_fit(
        t_fit, bs_samples_fit, cov_matrix, T, initial_params, bounds
    )

    A0_list = [params["A0"] for params in fitted_params_list if params is not None]
    m0_list = [params["m0"] for params in fitted_params_list if params is not None]
    A1_list = [params["A1"] for params in fitted_params_list if params is not None]
    m1_list = [params["m1"] for params in fitted_params_list if params is not None]

    print(
        "Number of successful two-state fits:",
        sum(success_list),
        "out of",
        len(success_list),
    )

    result_dict = {
        "A0_list": A0_list,
        "m0_list": m0_list,
        "A1_list": A1_list,
        "m1_list": m1_list,
        "success_list": success_list,
        "chi2_list": chi2_list,
        "redchi2_list": redchi2_list,
        "tmin": tmin,
        "tmax": tmax,
        "cov_mode": cov_mode,
    }

    if savepath is not None:
        np.savez(savepath, **result_dict)
        print(f"Saved correlated two-state fit results to {savepath}")

    return result_dict


# ----------------------------------------------------------------------
#  扫描 tmin 的封装（仍然可以直接调用）
# ----------------------------------------------------------------------
def scan_tmin_two_state(
    path,
    T,
    tmin_start=1,
    tmin_end=24,
    tmax=30,
    outdir="./data/processed/mom/bs_fit_results/scan_two_state_corr",
    cov_mode="full",
    eps=1e-12,
):
    """
    扫描 tmin (固定 tmax) 进行 two-state correlated bootstrap fit。

    对每个 tmin 产生文件：
       two_state_corr_tmin_{tmin}_tmax_{tmax}.npz
    """

    os.makedirs(outdir, exist_ok=True)

    print(f"[Two-state Correlated Scan] tmin={tmin_start} → {tmin_end} (tmax={tmax})")
    print("-" * 60)

    for tmin in range(tmin_start, tmin_end + 1):

        savefile = f"{outdir}/two_state_corr_tmin_{tmin}_tmax_{tmax}.npz"

        print(
            f"Running two-state correlated fit: tmin={tmin}, tmax={tmax}, cov_mode={cov_mode}"
        )

        result_dict = run_two_state_fit(
            path=path,
            T=T,
            tmin=tmin,
            tmax=tmax,
            savepath=savefile,
            cov_mode=cov_mode,
            eps=eps,
        )

        avg_redchi2 = np.mean(result_dict["redchi2_list"])
        print(f"Saved: {savefile}, avg redchi2 = {avg_redchi2:.5f}")

    print("Two-state correlated scan finished.\n")


def scan_tmin_one_state(
    path,
    T,
    tmin_start=1,
    tmin_end=24,
    tmax=30,
    outdir="./data/processed/mom/bs_fit_results/scan_one_state_corr",
    cov_mode="full",
    eps=1e-12,
):
    """
    扫描 tmin (固定 tmax) 进行 one-state correlated bootstrap fit。

    对每个 tmin 产生文件：
       one_state_corr_tmin_{tmin}_tmax_{tmax}.npz
    """

    os.makedirs(outdir, exist_ok=True)

    print(f"[One-state Correlated Scan] tmin={tmin_start} → {tmin_end} (tmax={tmax})")
    print("-" * 60)

    for tmin in range(tmin_start, tmin_end + 1):

        savefile = f"{outdir}/one_state_corr_tmin_{tmin}_tmax_{tmax}.npz"

        print(
            f"Running one-state correlated fit: tmin={tmin}, tmax={tmax}, cov_mode={cov_mode}"
        )

        result_dict = run_one_state_fit(
            path=path,
            T=T,
            tmin=tmin,
            tmax=tmax,
            savepath=savefile,
            cov_mode=cov_mode,
            eps=eps,
        )

        avg_redchi2 = np.mean(result_dict["redchi2_list"])
        print(f"Saved: {savefile}, avg redchi2 = {avg_redchi2:.5f}")

    print("One-state correlated scan finished.\n")


# ----------------------------------------------------------------------
#  AICc / m0 / plateau 等工具函数（几乎原样保留）
# ----------------------------------------------------------------------
def get_redchi2_nmin(path):
    filelist = sorted(glob.glob(os.path.join(path, "two_state*_tmin_*_tmax_*.npz")))
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
    """
    filelist = sorted(glob.glob(os.path.join(path, "*_tmin_*_tmax_*.npz")))
    for filepath in filelist:
        data = np.load(filepath)
        redchi2_vals = data["redchi2_list"]

        basename = os.path.basename(filepath).replace(".npz", "")
        parts = basename.split("_")

        try:
            i_tmin = parts.index("tmin")
            tmin = int(parts[i_tmin + 1])
            i_tmax = parts.index("tmax")
            tmax = int(parts[i_tmax + 1])
        except ValueError:
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

    在以 chi2 = r^T C^{-1} r 定义的情形下，常用形式：
        AIC  = chi2 + 2*k
        AICc = chi2 + 2*k + 2*k*(k+1)/(n - k - 1)
    """
    aic = chi2 + 2 * n_param
    correction = (2 * n_param * (n_param + 1)) / (n_data - n_param - 1)
    return aic + correction


def get_bootstrap_aicc(path):
    """
    读取指定目录下的 *_tmin_*_tmax_*.npz 文件，利用其中的 chi2_list
    计算各个 tmin 的平均 AICc。
    """
    filelist = sorted(glob.glob(os.path.join(path, "*_tmin_*_tmax_*.npz")))
    aicc_results = {}

    for filepath in filelist:
        data = np.load(filepath)
        chi2_vals = data["chi2_list"]

        basename = os.path.basename(filepath).replace(".npz", "")
        parts = basename.split("_")

        if "one_state" in basename:
            n_param = 2
        elif "two_state" in basename:
            n_param = 4
        else:
            raise ValueError(f"Cannot infer n_param from filename: {basename}")

        try:
            i_tmin = parts.index("tmin")
            tmin = int(parts[i_tmin + 1])
            i_tmax = parts.index("tmax")
            tmax = int(parts[i_tmax + 1])
        except ValueError:
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
    """
    filelist = sorted(glob.glob(os.path.join(path, "*_tmin_*_tmax_*.npz")))
    m0_results = {}

    for filepath in filelist:
        data = np.load(filepath)
        m0_vals = data["m0_list"]

        basename = os.path.basename(filepath).replace(".npz", "")
        parts = basename.split("_")

        try:
            i_tmin = parts.index("tmin")
            tmin = int(parts[i_tmin + 1])
        except ValueError:
            tmin = int(parts[3])

        m0_mean = np.mean(m0_vals)
        m0_error = np.std(m0_vals, ddof=1)
        m0_results[tmin] = (m0_mean, m0_error)

    return m0_results


def select_best_fit_by_aicc(one_state_dir, two_state_dir):
    """
    读取一态与二态目录中所有 tmin 对应的 AICc，
    对每个 tmin 比较两种模型的 AICc。
    """
    aicc_one = get_bootstrap_aicc(one_state_dir)
    aicc_two = get_bootstrap_aicc(two_state_dir)

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
    """
    m_plateau = m_vals[n_lower : n_upper + 1]
    err_plateau = m_errs[n_lower : n_upper + 1]

    partA = 1.0 / (err_plateau**2)
    averaged_m = np.sum(m_plateau * partA) / np.sum(partA)
    stat_err = np.sqrt(1.0 / np.sum(partA))
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
    画出 plateau 平均图（与原文件一致）。
    """
    m_avg, stat_err, sys_err = weighted_plateau_average(
        n_lower, n_upper, m_vals, m_errs
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.0))

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

    x_min = np.min(n_list) - 0.5
    x_max = np.max(n_list) + 0.5

    ax.fill_between(
        [x_min, x_max],
        m_avg - sys_err,
        m_avg + sys_err,
        color="red",
        alpha=0.3,
        label="syst",
    )

    ax.fill_between(
        [x_min, x_max],
        m_avg - stat_err,
        m_avg + stat_err,
        color="blue",
        alpha=0.5,
        label="stat",
    )

    ax.axhline(m_avg, color="blue", linewidth=1.2)

    ax.axvspan(
        n_list[n_lower] - 0.5,
        n_list[n_upper] + 0.5,
        color="grey",
        alpha=0.1,
        zorder=0,
    )

    ax.set_title("red: syst, blue: stat", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300)
    plt.show()

    print(f"plateau range: indices [{n_lower}, {n_upper}]")
    print(f"<M> = {m_avg:.6f}")
    print(f"stat err = {stat_err:.6f}")
    print(f"syst err = {sys_err:.6f}")

    return m_avg, stat_err, sys_err


def gaussian_plateau_averaging(
    tmin_vals, m_vals, m_errs, plateau_range, n_samples=5000, plot=True
):
    """
    Gaussian plateau averaging (与原文件一致)。
    """
    tmin_left, tmin_right = plateau_range

    mask = [(t >= tmin_left) and (t <= tmin_right) for t in tmin_vals]
    m_plateau = m_vals[mask]
    err_plateau = m_errs[mask]
    t_plateau = np.array(tmin_vals)[mask]

    combined_samples = []
    for i in range(len(m_plateau)):
        samples_i = np.random.normal(
            loc=m_plateau[i], scale=err_plateau[i], size=n_samples
        )
        combined_samples.append(samples_i)

    combined_samples = np.array(combined_samples)
    merged = combined_samples.flatten()

    median = np.median(merged)
    low = np.percentile(merged, 16)
    high = np.percentile(merged, 84)
    err_minus = median - low
    err_plus = high - median

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    # 这里不做实际运行，只留作简单示例/自测接口。
    # 你可以在 main.py 中 import 本模块并调用 run_one_state_fit / run_two_state_fit。
    pass
