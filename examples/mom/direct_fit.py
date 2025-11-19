"""
Description: one state and two state direct fit
@author: George
@since: 2025.11.19
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib as mpl


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

    _, redchi2, _ = compute_chi2_red(result, len(y_fit), len(initial_params))

    fitted_params = {"A0": result.x[0], "m0": result.x[1]}
    return fitted_params, result.cost, result.message, redchi2


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

    _, redchi2, _ = compute_chi2_red(result, len(y_fit), len(initial_params))

    fitted_params = {
        "A0": result.x[0],
        "m0": result.x[1],
        "A1": result.x[2],
        "m1": result.x[3],
    }
    return fitted_params, result.cost, result.message, redchi2


def load_p2_data(path):
    data = np.loadtxt(path)
    y_mean = data[:, 0]
    y_err = data[:, 1]
    return y_mean, y_err


def compute_chi2_red(result, n_data, n_params):
    chi2 = 2 * result.cost
    dof = n_data - n_params
    redchi2 = chi2 / dof
    return chi2, redchi2, dof


def result_plot(t, y_mean, y_err, fitted_params, T, state="one"):
    mpl.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 13,
            "figure.titlesize": 18,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",  # 更接近 LaTeX 的科研风字体
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    def plot_one_cosh_fit(t, y_mean, y_err, fitted_params, T):
        plt.figure(figsize=(8, 5))

        # --- 数据点 ---
        plt.errorbar(
            t,
            y_mean,
            y_err,
            fmt="o",
            markersize=4,
            capsize=3,
            label=r"$G(\tau),\; \hat{p}^2=0$",
        )

        # --- 拟合曲线 ---
        t_plot = np.linspace(0, T // 2, 300)
        y_plot = one_cosh_func([fitted_params["A0"], fitted_params["m0"]], t_plot, T)
        plt.plot(
            t_plot,
            y_plot,
            label=r"Fit: $C(\tau)=A_0\cosh[m_0(\tau-T/2)]$",
            linewidth=2.0,
            color="red",
        )

        # --- 坐标轴 ---
        plt.xlabel(r"$\tau/a$")
        plt.ylabel(r"$G(\tau)$")
        plt.yscale("log")
        # plt.xlim(0, T // 2)

        # --- 图注（科研风格） ---
        caption = (
            rf"$A_0 = {fitted_params['A0']:.3e}$, "
            rf"$am_0 = {fitted_params['m0']:.5f}$"
        )

        plt.figtext(0.5, -0.05, caption, ha="center", fontsize=13)

        # --- 标题 & legend ---
        plt.title(r"One-State Direct Fit, $\hat{p}^2=0$")
        plt.legend()

        plt.tight_layout(rect=[0, 0.06, 1, 1])  # 为 caption 留空间
        plt.show()

    # plot_one_cosh_fit(t, y_mean, y_err, fitted_params, T)

    def plot_two_cosh_fit(t, y_mean, y_err, fitted_params, T):
        plt.figure(figsize=(8, 5))

        # --- 数据点 ---
        plt.errorbar(
            t,
            y_mean,
            y_err,
            fmt="o",
            markersize=4,
            capsize=3,
            label=r"$G(\tau),\; \hat{p}^2=0$",
        )

        # --- 拟合曲线 ---
        t_plot = np.linspace(0, T // 2, 300)
        y_plot = two_cosh_func(
            [
                fitted_params["A0"],
                fitted_params["m0"],
                fitted_params["A1"],
                fitted_params["m1"],
            ],
            t_plot,
            T,
        )
        plt.plot(
            t_plot,
            y_plot,
            label=r"Fit: $C(\tau)=A_0\cosh[m_0(\tau-T/2)]+A_1\cosh[m_1(\tau-T/2)]$",
            linewidth=2.0,
            color="red",
        )

        # --- 坐标轴 ---
        plt.xlabel(r"$\tau/a$")
        plt.ylabel(r"$G(\tau)$")
        plt.yscale("log")

        # --- 标题 & legend ---
        plt.title(r"Two-State Direct Fit, $\hat{p}^2=0$")
        plt.legend()

        plt.tight_layout()
        plt.show()

    if state == "one":
        plot_one_cosh_fit(t, y_mean, y_err, fitted_params, T)
    elif state == "two":
        plot_two_cosh_fit(t, y_mean, y_err, fitted_params, T)


def run_one_state_fit(
    path="./data/processed/mom/p2_bs_mean_err/phi_p2_0_mean_err.dat",
    T=96,
    tmin=5,
    tmax=16,
):
    y_mean, y_err = load_p2_data(path)
    T = T
    t = np.arange(len(y_mean))

    tmin = tmin
    tmax = tmax
    mask = (t >= tmin) & (t <= tmax)
    t_fit = t[mask]
    y_fit = y_mean[mask]
    y_err_fit = y_err[mask]

    initial_params = {"A0": 1e-13, "m0": 0.6}
    bounds = ([0, 0.5], [1, 0.7])
    fitted_params, cost, message, redchi2 = one_state_direct_fit(
        t_fit, y_fit, y_err_fit, T, initial_params, bounds
    )
    print("Fitted Parameters:", fitted_params)
    print("Cost:", cost)
    print("Message:", message)
    print("Reduced Chi-squared:", redchi2)

    result_plot(t, y_mean, y_err, fitted_params, T)


def run_two_state_fit(
    path="./data/processed/mom/p2_bs_mean_err/phi_p2_0_mean_err.dat",
    T=96,
    tmin=5,
    tmax=16,
):
    y_mean, y_err = load_p2_data(path)
    T = T
    t = np.arange(len(y_mean))

    tmin = tmin
    tmax = tmax
    mask = (t >= tmin) & (t <= tmax)
    t_fit = t[mask]
    y_fit = y_mean[mask]
    y_err_fit = y_err[mask]

    initial_params = {"A0": 1e-15, "m0": 0.62, "A1": 1e-17, "m1": 1}
    bounds = ([0, 0.55, 0, 0.8], [1e-14, 0.65, 1e-15, 1.4])
    fitted_params, cost, message, redchi2 = two_state_direct_fit(
        t_fit, y_fit, y_err_fit, T, initial_params, bounds
    )
    print("Fitted Parameters:", fitted_params)
    print("Cost:", cost)
    print("Message:", message)
    print("Reduced Chi-squared:", redchi2)

    # result_plot(t, y_mean, y_err, fitted_params, T)
    result_plot(t, y_mean, y_err, fitted_params, T, state="two")


if __name__ == "__main__":

    path = "./data/processed/mom/p2_bs_mean_err/phi_p2_0_mean_err.dat"

    # run_one_state_fit(path=path, T=96, tmin=5, tmax=16)
    run_two_state_fit(path=path, T=96, tmin=5, tmax=30)

    # plt.figure(figsize=(8, 5))
    # plt.errorbar(t, y_mean, y_err, fmt="o", label=r"$data\ \hat{p}^2=0$", markersize=4)
    # t_plot = np.linspace(0, 48, 100)
    # y_plot = one_cosh_func([fitted_params["A0"], fitted_params["m0"]], t_plot, T)
    # plt.plot(t_plot, y_plot, label="Fitted One-Cosh", color="red")
    # plt.xlabel(r"$\tau/a$")
    # plt.ylabel(r"$G(\tau)$")
    # plt.yscale("log")
    # plt.figtext(
    #     0.15,
    #     0.8,
    #     rf"A0 = {fitted_params['A0']:.3e}\n $am_0$ = {fitted_params['m0']:.5f}\n",
    #     bbox=dict(facecolor="white", alpha=0.5),
    # )
    # plt.title("One-State Direct Fit, p^2=0")
    # plt.legend()
    # plt.grid()
    # plt.show()
