"""
Description: load dat files of all configurations,sorted by p^2,save to npy files
@author: George Liu
@since: 2025.11
"""

import numpy as np
from collections import defaultdict
import os
from glob import glob


def load_single_data(file_name):
    """
    读取单个 TwoPt 配置文件（一个 config）。

    返回:
        data_dict[(px,py,pz,mumu)] = C(t) (real part)
    """

    # columns: px py pz mumu t real imag
    data = np.loadtxt(file_name, comments="#")

    data_dict = {}

    # 获取所有出现过的 (px,py,pz,mumu)
    channels = np.unique(data[:, :4], axis=0)  # 前4列: px py pz mumu

    for px, py, pz, mumu in channels:
        mask = (
            (data[:, 0] == px)
            & (data[:, 1] == py)
            & (data[:, 2] == pz)
            & (data[:, 3] == mumu)
        )

        sub = data[mask]
        sub_sorted = sub[np.argsort(sub[:, 4])]  # sort by t

        real_part = sub_sorted[:, 5]

        data_dict[(int(px), int(py), int(pz), int(mumu))] = real_part

    return data_dict


def bin_by_p2(data_dict):
    """
    输入:
        data_dict[(px,py,pz,mumu)] = C(t)

    输出:
        p2_dict[p2][mumu] = [C(t), C'(t), ...]
    """
    p2_dict = defaultdict(lambda: defaultdict(list))

    for (px, py, pz, mumu), C in data_dict.items():
        p2 = px * px + py * py + pz * pz
        p2_dict[p2][mumu].append(C)

    return p2_dict


def average_momenta_in_p2_bin(p2_dict):
    """
    输入:
        p2_dict[p2][mumu] = [C1(t), C2(t), ...]

    输出:
        avg_dict[p2][mumu] = averaged C(t)
    """

    avg_dict = defaultdict(dict)

    for p2, mumu_dict in p2_dict.items():
        for mumu, C_list in mumu_dict.items():
            C_stack = np.vstack(C_list)
            C_avg = np.mean(C_stack, axis=0)
            avg_dict[p2][mumu] = C_avg

    return avg_dict


def average_mumu_phi(avg_dict):
    """
    输入：
        avg_dict[p2][mumu] = C(t)

    输出：
        phi_dict[p2] = C_phi(t)
    """

    phi_dict = {}

    for p2, mumu_dict in avg_dict.items():
        spatial = []
        for mu in [0, 1, 2]:
            if mu in mumu_dict:
                spatial.append(mumu_dict[mu])

        if len(spatial) == 0:
            continue  # 该 p2 没有 spatial μμ

        # 平均 spatial 组件
        phi_C = np.mean(np.vstack(spatial), axis=0)
        phi_dict[p2] = phi_C

    return phi_dict


def fold_single_correlator(C, T):
    """
    折叠单个时间相关函数 C(t)。
    输入:
        C: 长度为 T 的时间相关函数数组
        T: 时间方向长度
    输出:
        Cf: 折叠后的时间相关函数，长度为 T/2 + 1
    """
    T = int(T)
    half = T // 2
    Cf = np.zeros(half + 1)

    for t in range(half):
        Cf[t] = 0.5 * (C[t] + C[T - t - 1])
    Cf[half] = C[half]

    return Cf


def save_p2_correlators(phi_dict, output_file):
    """
    保存 per-config 的 (p², C(t)) 数据到文本文件。

    格式：
        p2  t  C(t)
    """

    with open(output_file, "w") as f:
        f.write("# Correlator file (phi meson, p²-binned, single config)\n")
        f.write("# p2   time   correlator_real\n")
        for p2, C in sorted(phi_dict.items()):
            T = len(C)
            for t in range(T):
                f.write(f"{p2} {t} {C[t]:.16e}\n")


def process_all_dat(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(input_dir, "TwoPt*.dat")))
    print(f"Found {len(files)} config files.")

    for fpath in files:
        conf_name = os.path.basename(fpath).replace(".dat", "")
        outpath = os.path.join(output_dir, f"{conf_name}_p2.txt")

        print(f"Processing {conf_name} ...")

        # pipeline
        data_dict = load_single_data(fpath)
        p2_dict = bin_by_p2(data_dict)
        avg_dict = average_momenta_in_p2_bin(p2_dict)
        phi_dict = average_mumu_phi(avg_dict)
        folded_phi_dict = {
            p2: fold_single_correlator(C, T=96) for p2, C in phi_dict.items()
        }

        save_p2_correlators(folded_phi_dict, outpath)


if __name__ == "__main__":

    process_all_dat(
        input_dir="/home/george/Documents/WorkSpace/Lattice/corr_fit/data/raw/s3_mom",
        output_dir="/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/p2_sorted/",
    )
