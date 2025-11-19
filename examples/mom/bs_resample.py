"""
Description: load all p2_sorted data and use bootstrap
 to get the mean & error of all different p2
@author: George Liu
@since: 2025.11
"""

import os
import numpy as np
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from latqcdtools.statistics.bootstr import bootstr


# ============================================================
# 0. 你的 bootstrap_resample 函数
# ============================================================
def bootstrap_resample(data, n_resamples=1000, return_samples=True, random_state=42):
    """
    自定义 Bootstrap 重采样函数

    参数
    ----
    data: shape (N_conf, T)
    n_resamples: 重采样次数
    return_samples: True → 返回所有 bootstrap 样本
    random_state: 随机种子

    返回
    ----
    samples: shape (n_resamples, T)
    """
    rng = np.random.default_rng(random_state)
    N, T = data.shape

    # bootstrap 索引矩阵 (n_resamples, N_conf)
    idx = rng.integers(0, N, size=(n_resamples, N))

    # 每次重采样后在 config 方向均值
    samples = np.mean(data[idx], axis=1)

    return samples  # shape = (n_resamples, T)


# ============================================================
# 1. 读取单个 config 的 p² 处理文件
# ============================================================
def load_p2_file(fname):
    data = np.loadtxt(fname, comments="#")
    p2_list = np.unique(data[:, 0]).astype(int)

    p2_dict = {}
    for p2 in p2_list:
        mask = data[:, 0] == p2
        sub = data[mask]
        sub = sub[np.argsort(sub[:, 1])]
        p2_dict[p2] = sub[:, 2]  # correlator

    return p2_dict


# ============================================================
# 2. 构建 ensemble[p2] → (N_conf, T)
# ============================================================
def build_ensemble(file_list):
    print(f"Building ensemble from {len(file_list)} configs...")

    temp = defaultdict(list)

    for fname in file_list:
        p2_dict = load_p2_file(fname)
        for p2, C in p2_dict.items():
            temp[p2].append(C)

    ensemble = {}
    for p2, C_list in temp.items():
        arr = np.vstack(C_list)
        ensemble[p2] = arr
        print(f"p2={p2}: shape={arr.shape} (N_conf, T)")

    return ensemble


# ============================================================
# 3. 对 ensemble 进行 bootstrap → 使用你提供的 bootstrap_resample
# 返回 boot_dict[p2] = (N_bs, T)
# ============================================================
def bootstrap_ensemble(ensemble, Nbs=1000):
    boot_dict = {}

    for p2, arr in ensemble.items():
        # boots, *_ = bootstrap_resample(arr, n_resamples=Nbs, return_samples=True)
        boots, *_ = bootstr(
            func=my_func, data=arr, numb_samples=Nbs, return_sample=True, conf_axis=0
        )
        boot_dict[p2] = np.array(boots)

        print(f"Bootstrap p2={p2}: {arr.shape[0]} configs → {Nbs} samples")

    return boot_dict


# ============================================================
# 4. 保存 bootstrap 样本
# ============================================================
# def save_bootstrap_results(boot_dict, outdir):
#     import os

#     # Nbs, T = list(boot_dict.values())[1].shape

#     os.makedirs(outdir, exist_ok=True)

#     for p2, boots in boot_dict.items():
#         Nbs, T = boots.shape

#         outfile = f"{outdir}/phi_p2_{p2}_bootstrap.dat"
#         with open(outfile, "w") as f:
#             f.write("# Bootstrap phi meson correlators\n")
#             f.write(f"# p2 = {p2}, Nbs = {Nbs}, T = {T}\n")
#             f.write("# sample   time   correlator\n")

#             for b in range(Nbs):
#                 for t in range(T):
#                     f.write(f"{b}  {t}  {boots[b, t]:.16e}\n")

#         print(f"Saved: {outfile}")


def save_bootstrap_results(boot_dict, outdir):
    """
    保存 bootstrap 样本为 npy 文件，避免庞大的 txt 文件；
    另外生成一个 meta 信息文件，供检查使用。

    boot_dict: {p2: ndarray(Nbs, T)}
    outdir   : 输出目录
    """

    os.makedirs(outdir, exist_ok=True)

    for p2, boots in boot_dict.items():
        Nbs, T = boots.shape

        # ------------------------------
        # 1) 保存为 .npy （最佳格式）
        # ------------------------------
        npy_out = f"{outdir}/phi_p2_{p2}_bs.npy"
        np.save(npy_out, boots)

        # ------------------------------
        # 2) 保存 meta 信息（可选）
        # ------------------------------
        # meta_out = f"{outdir}/phi_p2_{p2}_bootstrap.meta"
        # with open(meta_out, "w") as f:
        #     f.write("# Bootstrap phi meson correlators metadata\n")
        #     f.write(f"p2 = {p2}\n")
        #     f.write(f"Nbs = {Nbs}\n")
        #     f.write(f"T = {T}\n")
        #     f.write("format = numpy .npy (Nbs, T)\n")

        print(f"Saved samples: {npy_out}")
        # print(f"Saved meta   : {meta_out}")


# ============================================================
# 5. 一键 pipeline
# ============================================================
def run_pipeline(indir="configs_p2", outdir="bootstrap_output", Nbs=10):
    file_list = sorted(glob(f"{indir}/*.txt"))

    ensemble = build_ensemble(file_list)

    boot_dict = bootstrap_ensemble(ensemble, Nbs=Nbs)
    print(boot_dict[0].shape)

    save_bootstrap_results(boot_dict, outdir)

    print("\n=== Pipeline finished successfully! ===")


def my_func(data):
    return np.mean(data, axis=0)


def save_data(input_folder, pattern="*.npy"):
    files = sorted(glob.glob(os.path.join(input_folder, pattern)))

    for f in files:
        # 自动解析 p2
        basename = os.path.basename(f)  # temp_p2_3_bs.npy
        # 提取 3 -> int
        p2 = int(basename.split("_")[2])

        samples = np.load(f)  # shape (Nbs, T)

        mean = np.mean(samples, axis=0)
        err = np.std(samples, axis=0, ddof=1)

        data = np.column_stack([mean, err])  # shape (T, 2)
        header = "# mean err"
        np.savetxt(
            f"/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/p2_bs_mean_err/phi_p2_{p2}_mean_err.dat",
            data,
            header=header,
        )


# ============================================================
# 调用示例
# ============================================================
if __name__ == "__main__":
    run_pipeline(
        indir="/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/p2_sorted/",
        outdir="/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_samples/",
        Nbs=1000,
    )
