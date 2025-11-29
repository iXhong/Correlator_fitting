import numpy as np
import matplotlib.pyplot as plt
from bootstrap_correlated_fit import run_one_state_fit, run_two_state_fit
from bootstrap_fit_fixed import one_cosh_func, two_cosh_func


def test_one_state_fit_correlated(path, T, tmin, tmax, cov_mode, outpath):
    result = run_one_state_fit(
        path=path, T=T, tmin=tmin, tmax=tmax, cov_mode=cov_mode, savepath=outpath
    )
    m0_list = result["m0_list"]
    A0_list = result["A0_list"]
    m0_mean = np.mean(m0_list)
    m0_std = np.std(m0_list, ddof=1)
    A0_mean = np.mean(A0_list)
    A0_std = np.std(A0_list, ddof=1)
    print(f"One-state fit (correlated) results for tmin={tmin}, tmax={tmax}:")
    print(f"m0 = {m0_mean:.6f} ± {m0_std:.6f}")
    print(f"A0 = {A0_mean} ± {A0_std}")


def test_two_state_fit_correlated(path, T, tmin, tmax, cov_mode, outpath):
    result = run_two_state_fit(
        path=path, T=T, tmin=tmin, tmax=tmax, cov_mode=cov_mode, savepath=outpath
    )
    m0_list = result["m0_list"]
    A0_list = result["A0_list"]
    m1_list = result["m1_list"]
    A1_list = result["A1_list"]
    m0_mean = np.mean(m0_list)
    m0_std = np.std(m0_list, ddof=1)
    A0_mean = np.mean(A0_list)
    A0_std = np.std(A0_list, ddof=1)
    m1_mean = np.mean(m1_list)
    m1_std = np.std(m1_list, ddof=1)
    A1_mean = np.mean(A1_list)
    A1_std = np.std(A1_list, ddof=1)
    print(f"Two-state fit (correlated) results for tmin={tmin}, tmax={tmax}:")
    print(f"m0 = {m0_mean:.6f} ± {m0_std:.6f}")
    print(f"A0 = {A0_mean} ± {A0_std}")
    print(f"m1 = {m1_mean:.6f} ± {m1_std:.6f}")
    print(f"A1 = {A1_mean} ± {A1_std}")
    return result


def plot_bootstrap_fit_result(result_dict_one, result_dict_two):

    # result_dict_one = np.load(outpath1)
    # result_dict_two = np.load(outpath2)
    A0_list = result_dict_two["A0_list"]
    m0_list = result_dict_two["m0_list"]
    A1_list = result_dict_two["A1_list"]
    m1_list = result_dict_two["m1_list"]

    y_fit_samples = []
    y_fit_samples1 = []
    t = np.arange(49)
    for i in range(len(A0_list)):
        params = {
            "A0": A0_list[i],
            "m0": m0_list[i],
            "A1": A1_list[i],
            "m1": m1_list[i],
        }
        y_fit = two_cosh_func(
            [params["A0"], params["m0"], params["A1"], params["m1"]], t, T=96
        )
        y_fit_samples.append(y_fit)
    y_fit_samples = np.array(y_fit_samples)
    y_fit_mean = np.mean(y_fit_samples, axis=0)
    y_fit_std = np.std(y_fit_samples, axis=0, ddof=1)
    for i in range(len(A0_list)):
        params = {
            "A0": result_dict_one["A0_list"][i],
            "m0": result_dict_one["m0_list"][i],
        }
        y_fit1 = one_cosh_func([params["A0"], params["m0"]], t, T=96)
        y_fit_samples1.append(y_fit1)
    y_fit_samples1 = np.array(y_fit_samples1)
    y_fit_mean1 = np.mean(y_fit_samples1, axis=0)
    y_fit_std1 = np.std(y_fit_samples1, axis=0, ddof=1)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        t,
        y_fit_mean,
        y_fit_std,
        fmt="+",
        label=r"$two\ cosh\ Fitted\ Curve$",
        markersize=6,
    )
    plt.errorbar(
        t,
        y_fit_mean1,
        y_fit_std1,
        fmt="x",
        label=r"$one\ cosh\ Fitted\ Curve$",
        markersize=6,
    )
    y_data = np.loadtxt(
        "./data/processed/mom/p2_bs_mean_err/phi_p2_0_mean_err.dat", comments="#"
    )[:, 0]
    y_err = np.loadtxt(
        "./data/processed/mom/p2_bs_mean_err/phi_p2_0_mean_err.dat", comments="#"
    )[:, 1]

    plt.errorbar(
        np.arange(len(y_data)),
        y_data,
        y_err,
        fmt="o",
        label=r"$Data\ \hat{p}^2=0$",
        markersize=4,
    )

    plt.xlabel(r"$\tau$/a")
    plt.ylabel(r"$a^3$G($\tau$)")
    plt.yscale("log")
    plt.title("Bootstrap Fit Result")
    plt.legend()
    plt.grid()
    plt.show()


def plot_result_dict_two(result_dict_two):

    A0_list = result_dict_two["A0_list"]
    m0_list = result_dict_two["m0_list"]
    A1_list = result_dict_two["A1_list"]
    m1_list = result_dict_two["m1_list"]

    y_fit_samples = []
    t = np.arange(49)
    for i in range(len(A0_list)):
        params = {
            "A0": A0_list[i],
            "m0": m0_list[i],
            "A1": A1_list[i],
            "m1": m1_list[i],
        }
        y_fit = two_cosh_func(
            [params["A0"], params["m0"], params["A1"], params["m1"]], t, T=96
        )
        y_fit_samples.append(y_fit)
    y_fit_samples = np.array(y_fit_samples)
    y_fit_mean = np.mean(y_fit_samples, axis=0)
    y_fit_std = np.std(y_fit_samples, axis=0, ddof=1)
    y_data = np.loadtxt(
        "./data/processed/mom/p2_bs_mean_err/phi_p2_0_mean_err.dat", comments="#"
    )[:, 0]
    y_err = np.loadtxt(
        "./data/processed/mom/p2_bs_mean_err/phi_p2_0_mean_err.dat", comments="#"
    )[:, 1]

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        t,
        y_fit_mean,
        y_fit_std,
        fmt="+",
        label=r"$two\ cosh\ Fitted\ Curve$",
        markersize=6,
    )

    plt.errorbar(
        np.arange(len(y_data)),
        y_data,
        y_err,
        fmt="o",
        label=r"$Data\ \hat{p}^2=0$",
        markersize=4,
    )
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$a^3$G($\tau$)")
    plt.yscale("log")
    plt.title("Bootstrap Fit Result")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    filepath = "./data/processed/mom/bs_samples/phi_p2_0_bs.npy"

    result_dict_two = test_two_state_fit_correlated(
        path=filepath, T=96, tmin=4, tmax=30, cov_mode="full", outpath=None
    )

    plot_result_dict_two(result_dict_two)
