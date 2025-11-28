import glob
import os
import numpy as np
from matplotlib import pyplot as plt

from direct_fit import two_cosh_func, one_cosh_func


def load_bs_folder(folder):
    """
    读取形如:
        phi_p2_0_bs.npy
        phi_p2_1_bs.npy
        ...
    的 bootstrap 样本文件

    return:
        dict[p2] = samples (Nbs, T)
    """
    files = sorted(glob.glob(os.path.join(folder, "phi_p2_*_bs.npy")))

    bs_dict = {}

    for f in files:
        # 自动解析 p2
        basename = os.path.basename(f)  # phi_p2_3_bs.npy
        # 提取 3 -> int
        p2 = int(basename.split("_")[2])

        samples = np.load(f)  # shape (Nbs, T)
        bs_dict[p2] = samples

    return bs_dict


def compute_mean_error(bs_dict):
    """
    输入：dict[p2] = (Nbs, T)
    输出：
        mean_dict[p2] = 长度 T 的平均值
        err_dict[p2]  = 长度 T 的bootstrap误差
    """
    mean_dict = {}
    err_dict = {}

    for p2, samples in bs_dict.items():
        mean = np.mean(samples, axis=0)
        err = np.std(samples, axis=0, ddof=1)

        mean_dict[p2] = mean
        err_dict[p2] = err

    return mean_dict, err_dict


def generate_rainbow_colors(n, cmap_name="rainbow"):
    """
    生成 rainbow 风格的渐变颜色.
    cmap_name 可选：'rainbow', 'turbo', 'hsv', etc.
    """
    cmap = plt.get_cmap(cmap_name)
    # 从 0~1 均匀采样 n 个颜色
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_mean_error(mean_dict, err_dict):
    p2_list = sorted(mean_dict.keys())
    n = len(p2_list)
    colors = generate_rainbow_colors(n, cmap_name="turbo")  # Rainbow 风格

    plt.figure(figsize=(12, 8))

    for i, p2 in enumerate(p2_list):
        mean = mean_dict[p2]
        err = err_dict[p2]

        t = np.arange(len(mean))

        plt.errorbar(
            t,
            mean,
            yerr=err,
            fmt="o",
            capsize=3,
            color=colors[i],
            label=rf"$\hat p^2$={p2}",
        )

    plt.yscale("log")
    plt.xlabel(r"$n_t$")
    plt.ylabel(r"$C(p², n_t)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("corr_mean_error.png", dpi=300)
    plt.show()


def plot_bootstrap_fit_result():
    outpath1 = "./data/processed/mom/bs_fit_results/one_state_fit_results.npz"
    outpath3 = "./data/processed/mom/bs_fit_results/two_state_fit_results3.npz"

    # one state fit
    # result_dict_one = run_one_state_fit(
    #     path=inpath, T=96, tmin=5, tmax=20, savepath=outpath1
    # )

    # two state fit
    # result_dict_two = run_two_state_fit(
    #     path=inpath, T=96, tmin=5, tmax=20, savepath=outpath3
    # )
    result_dict_one = np.load(outpath1)
    result_dict_two = np.load(outpath3)
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

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$a^3$G($\tau$)")
    plt.yscale("log")
    plt.title("Bootstrap Fit Result")
    plt.legend()
    plt.grid()
    plt.show()


def plot_histogram(data, param_name, bins="auto"):
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel(param_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {param_name}")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # folder = "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_samples/"
    # bs_dict = load_bs_folder(folder)

    # mean_dict, err_dict = compute_mean_error(bs_dict)
    # mean, err = np.loadtxt(
    #     "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/p2_bs_mean_err/phi_p2_0_mean_err.dat",
    #     unpack=True,
    # )
    p2_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27]
    mean = []
    err = []
    for p2 in p2_list:
        m, e = np.loadtxt(
            f"/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/p2_bs_mean_err/phi_p2_{p2}_mean_err.dat",
            unpack=True,
        )
        mean.append(m)
        err.append(e)

    # plot_mean_error(mean_dict, err_dict)
    # plot all the p2 on one plot
    plt.figure(figsize=(12, 8))
    n = len(p2_list)
    colors = generate_rainbow_colors(n, cmap_name="turbo")  # Rainbow 风格
    for i, p2 in enumerate(p2_list):
        t = np.arange(len(mean[i]))
        plt.errorbar(
            t,
            mean[i],
            yerr=err[i],
            fmt="o",
            capsize=3,
            color=colors[i],
            label=rf"$\hat p^2$={p2}",
        )
    plt.yscale("log")
    plt.xlabel(r"$\tau/a$")
    plt.ylabel(r"$a^3G(\tau/a)$")
    plt.legend()
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.figtext(0.5, 0.01, r"$\hat{p}^2 = (a/2\pi)^2p^2$", ha="center", fontsize=10)
    # plt.savefig("corr_mean_error_all_p2.png", dpi=300)
    plt.show()
