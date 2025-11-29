#!/usr/bin/env python3
"""
main.py

Driver script for bootstrap cosh fits and AICc-based model selection.

把所有参数都写在文件顶部的“用户配置”区域，
直接 `python main.py` 跑即可。
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# from bootstrap_fit_fixed import (
#     scan_tmin_one_state,
#     scan_tmin_two_state,
#     get_bootstrap_m0,
#     select_best_fit_by_aicc,
#     get_bootstrap_m0,
#     get_best_m0_sequence,
#     plot_plateau,
#     weighted_plateau_average,
# )

from bootstrap_correlated_fit import (
    scan_tmin_one_state,
    scan_tmin_two_state,
    get_bootstrap_m0,
    select_best_fit_by_aicc,
    get_best_m0_sequence,
    plot_plateau,
    weighted_plateau_average,
)

# ===================== 用户配置区域 =====================

# 1. 输入：某个 p^2 的 bootstrap 样本文件（n_bs, T/2+1 或 T 等）
BS_FILE = "./data/processed/mom/bs_samples/phi_p2_0_bs.npy"

# 2. 格点时间长度 T
LATTICE_T = 96

# 3. 拟合区间扫描：tmin 从多少扫到多少，tmax 固定
TMIN_START = 1
TMIN_END = 24
TMAX_FIXED = 30

# 4. 拟合结果输出目录（分别存放 one-state / two-state 扫描）
ONE_STATE_DIR = "./data/processed/mom/bs_fit_results/scan_one_state_corr"
TWO_STATE_DIR = "./data/processed/mom/bs_fit_results/scan_two_state_corr"

# 5. plateau 区间（用 tmin 的值来表示）
PLATEAU_LEFT = 9  # 最左侧的 tmin
PLATEAU_RIGHT = 15  # 最右侧的 tmin

# 6. 是否画图
SHOW_PLOTS = True

# 7. Gaussian plateau averaging 样本数
N_GAUSS_SAMPLES = 5000

# ======================================================


def scan_tmin():
    # 确保输出目录存在
    os.makedirs(ONE_STATE_DIR, exist_ok=True)
    os.makedirs(TWO_STATE_DIR, exist_ok=True)

    print("=== Step 1: scan tmin for one-state fits ===")
    scan_tmin_one_state(
        path=BS_FILE,
        T=LATTICE_T,
        tmin_start=TMIN_START,
        tmin_end=TMIN_END,
        tmax=TMAX_FIXED,
        outdir=ONE_STATE_DIR,
    )

    print("=== Step 2: scan tmin for two-state fits ===")
    scan_tmin_two_state(
        path=BS_FILE,
        T=LATTICE_T,
        tmin_start=TMIN_START,
        tmin_end=TMIN_END,
        tmax=TMAX_FIXED,
        outdir=TWO_STATE_DIR,
    )


def plot_m0_selected():
    best_fit = select_best_fit_by_aicc(ONE_STATE_DIR, TWO_STATE_DIR)
    m0_one = get_bootstrap_m0(ONE_STATE_DIR)
    m0_two = get_bootstrap_m0(TWO_STATE_DIR)

    m0_one_means = [m0_one[t][0] for t in sorted(m0_one.keys())]
    m0_two_means = [m0_two[t][0] for t in sorted(m0_two.keys())]
    m0_one_errs = [m0_one[t][1] for t in sorted(m0_one.keys())]
    m0_two_errs = [m0_two[t][1] for t in sorted(m0_two.keys())]

    tmin_vals, m0_best, m0_err = get_best_m0_sequence(best_fit, m0_one, m0_two)

    # plot m0 of one state, two state and best selected
    tmin_vals = sorted(best_fit.keys())
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        tmin_vals,
        m0_one_means,
        yerr=m0_one_errs,
        fmt="o",
        capsize=3,
        label="one-state",
    )
    plt.errorbar(
        tmin_vals,
        m0_two_means,
        yerr=m0_two_errs,
        fmt="s",
        capsize=3,
        label="two-state",
    )
    plt.errorbar(
        tmin_vals,
        m0_best,
        yerr=m0_err,
        fmt="o",
        capsize=3,
        label="AICc selected",
    )
    plt.xlabel(r"$n_{\sigma,min}$")
    plt.ylabel(r"a$m_0$")
    plt.grid()
    plt.legend()
    plt.show()


def average_plateau_m0(tmin, tmax):
    best_fit = select_best_fit_by_aicc(ONE_STATE_DIR, TWO_STATE_DIR)
    m0_one = get_bootstrap_m0(ONE_STATE_DIR)
    m0_two = get_bootstrap_m0(TWO_STATE_DIR)

    tmin_vals, m0_best, m0_err = get_best_m0_sequence(best_fit, m0_one, m0_two)

    # plateau averaging
    # tmin = 5
    # tmax = 25

    n_lower = tmin_vals.index(tmin)
    n_upper = tmin_vals.index(tmax)
    print(n_lower, n_upper)
    m_avg, stat_err, sys_err = plot_plateau(
        n_list=tmin_vals,
        m_vals=m0_best,
        m_errs=m0_err,
        n_lower=n_lower,
        n_upper=n_upper,
        xlabel=r"$n_{\sigma,\min}$",
        ylabel=r"a$m_0$",
        outpath=None,
    )
    return m_avg, stat_err, sys_err


def convert_result(am0_mean, sys_err, stat_err):
    total_err = np.sqrt(sys_err**2 + stat_err**2)
    a = 0.117

    m0 = am0_mean / a * 197.3269804  # MeV
    m0_err = total_err / a * 197.3269804  # MeV
    return m0, m0_err


if __name__ == "__main__":

    # scan_tmin()
    # plot_m0_selected()
    m_avg, stat_err, sys_err = average_plateau_m0(tmin=4, tmax=24)
    m0, m0_err = convert_result(m_avg, stat_err, sys_err)
    print(f"Final m0 = {m0:.2f} ± {m0_err:.2f} MeV (stat+sys)")
