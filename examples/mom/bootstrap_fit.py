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


def prepare_p2_bs_samples(path):
    # sigma used for uncorrelated fitting
    bs_samples = np.load(path)  # shape (n_bootstrap, n_data)
    # sigma = np.std(bs_samples, axis=0, ddof=1) / (len(bs_samples))
    sigma = np.std(bs_samples, axis=0, ddof=1)

    return bs_samples, sigma


def run_one_state_fit(path, T, tmin, tmax, savepath=None):
    bs_samples, sigma = prepare_p2_bs_samples(path)
    T = 96
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
    T = 96
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


def add_chi2(path):
    filelist = sorted(glob.glob(os.path.join(path, "two_state_tmin_*_tmax_*.npz")))
    for filepath in filelist:
        data = np.load(filepath)
        redchi2_vals = data["redchi2_list"]
        chi2_vals = [
            redchi2 * (len(redchi2_vals) - 4) for redchi2 in redchi2_vals
        ]  # 4 parameters
        save_dict = {key: data[key] for key in data.files}
        save_dict["chi2_list"] = chi2_vals

        np.savez(filepath, **save_dict)
        print(f"Updated: {os.path.basename(filepath)}")
    print("all update")


def get_aicc(chi2, n_data, n_param):
    """
    Calculate the corrected Akaike Information Criterion (AICc).
    AICc = -2ln(chi2) + 2*n_param + (2*n_param*(n_param+1)) / (n_data - n_param - 1)
    """
    aic = 2 * n_param - 2 * np.log(chi2)
    correction = (2 * n_param * (n_param + 1)) / (n_data - n_param - 1)
    aicc = aic + correction
    return aicc


def get_bootstrap_aicc(path):
    filelist = sorted(glob.glob(os.path.join(path, "two_state_tmin_*_tmax_*.npz")))
    aicc_results = {}
    for filepath in filelist:
        data = np.load(filepath)
        chi2_vals = data["chi2_list"]
        tmin_str = os.path.basename(filepath).split("_")[3]
        tmin = int(tmin_str)
        if tmin <= 24:
            n_data = 30 - tmin
            print(n_data)
            n_param = 4  # for two-state fit

            aicc_vals = [get_aicc(chi2, n_data, n_param) for chi2 in chi2_vals]
            avg_aicc = np.mean(aicc_vals)

            aicc_results[tmin] = avg_aicc
    return aicc_results


if __name__ == "__main__":
    inpath = "./data/processed/mom/bs_samples/phi_p2_0_bs.npy"
    outpath1 = "./data/processed/mom/bs_fit_results/one_state_fit_results1.npz"
    outpath2 = "./data/processed/mom/bs_fit_results/two_state_fit_results2.npz"
    outpath3 = "./data/processed/mom/bs_fit_results/two_state_fit_results3.npz"
    inpath2 = "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_fit_results/scan_two_state/two_state_tmin_5_tmax_30.npz"

    # result_dict = np.load(inpath2)
    # print(result_dict.files)
    # chi2_list = result_dict["chi2_list"]
    # redchi2_list = result_dict["redchi2_list"]
    # print("chi2:", np.mean(chi2_list))
    # print("redchi2:", np.mean(redchi2_list))

    # scan_tmin_two_state(
    #     path=inpath,
    #     T=96,
    #     tmin_start=1,
    #     tmin_end=24,
    #     tmax=30,
    #     outdir="./data/processed/mom/bs_fit_results/scan_two_state",
    # )

    # aicc = get_bootstrap_aicc(
    #     path="./data/processed/mom/bs_fit_results/scan_one_state",
    # )
    # for tmin, aicc_val in aicc.items():
    #     print(f"tmin={tmin}, AICc={aicc_val:.2f}")
