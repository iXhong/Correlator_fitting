import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os


# -------------------------------
# 1) fold 单个 sample
# -------------------------------
def fold_single_correlator(C, T):
    T = int(T)
    half = T // 2
    Cf = np.zeros(half + 1)

    for t in range(half):
        Cf[t] = 0.5 * (C[t] + C[T - t - 1])
    Cf[half] = C[half]

    return Cf


# -------------------------------
# 2) fold 所有 samples
# -------------------------------
def fold_all_samples(data, T):
    return np.array([fold_single_correlator(d, T) for d in data])


# -------------------------------
# 3) 计算 mean & error
# -------------------------------
def bootstrap_mean_err(folded):
    mean = np.mean(folded, axis=0)
    err = np.std(folded, axis=0, ddof=1)
    return mean, err


# -------------------------------
# 4) 主程序：扫描目录 + 处理全部 p²
# -------------------------------
def process_all_p2(dirpath, Nbs=1000, T=96):
    """
    扫描目录 dirpath 下的所有 bootstrap dat 文件，
    自动读取 p² ，返回 {p2: (mean, err)} 字典
    """
    file_list = sorted(glob.glob(os.path.join(dirpath, "*.dat")))
    results = {}

    for fname in file_list:
        # 自动从文件名提取 p2
        # 如：phi_p2_0_bootstrap.dat → p2 = 0
        m = re.search(r"p2_(\d+)", fname)
        if not m:
            print(f"[跳过文件] 无法从文件名获取 p2: {fname}")
            continue
        p2 = int(m.group(1))

        print(f"正在处理 p2={p2}: {fname}")

        # 读取数据
        raw = np.loadtxt(fname)
        if raw.shape[1] < 3:
            raise ValueError(f"文件格式错误，需要至少三列: {fname}")

        # 第三列是 correlator
        corr = raw[:, 2]

        # reshape 成 (Nbs, T)
        data = corr.reshape(Nbs, T)

        # fold
        folded = fold_all_samples(data, T=T)

        # mean & err
        mean, err = bootstrap_mean_err(folded)

        # 保存结果
        results[p2] = (mean, err)

    return results


def save_results(results, outdir, T=96):
    """
    保存结果到 outdir，文件名格式：bs_folded_p2_{p2}_mean_err.dat
    """
    t_list = np.arange(0, (T // 2) + 1, dtype=int)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for p2, (mean, err) in results.items():
        outpath = os.path.join(outdir, f"bs_folded_p2_{p2}_mean_err.dat")
        data_to_save = np.vstack((mean, err)).T
        data_to_save = np.column_stack((t_list, data_to_save))
        np.savetxt(
            outpath,
            data_to_save,
            header="t bs_mean bs_err",
            fmt=["%d", "%.18e", "%.18e"],
        )
        print(f"Saved results for p2={p2} to {outpath}")


# -------------------------------
# 5) 绘图：所有 p² 的 errorbar
# -------------------------------
def plot_all_p2(results):
    """
    results: {p2: (mean, err)}
    """
    plt.figure(figsize=(9, 6))

    for p2, (mean, err) in sorted(results.items()):
        t = np.arange(len(mean))
        plt.errorbar(
            t,
            mean,
            yerr=err,
            fmt="o-",
            capsize=3,
            label=f"p² = {p2}",
            alpha=0.8,
        )

    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Correlator")
    plt.title("Bootstrap Folded Correlators for Different $p^2$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# ------------------------------------
# 6) 程序入口
# ------------------------------------
if __name__ == "__main__":
    dirpath = "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_resampled/"

    results = process_all_p2(dirpath, Nbs=1000, T=96)

    save_results(
        results,
        outdir="/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_folded/",
        T=96,
    )

    # plot_all_p2(results)
