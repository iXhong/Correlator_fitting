"""
I/O functions for correlator data
@author GeorgeLiu
@since 2025.11
"""

import numpy as np
import glob


def load_data(path, mumu):
    """
    load correlator data from .dat files
    Return:
        flip_data: folded (respecting periodic BC) & averaged data per config
        t: time data (0 .. T//2)
        N: number of correlator files
    """
    file_list = sorted(glob.glob(f"{path}/*.dat"))

    real_data_all = []

    for fname in file_list:
        data = np.loadtxt(fname, comments="#")
        filtered = data[data[:, 3] == mumu]  # 第4列是 mumu
        real_values = filtered[:, 5]
        real_data_all.append(real_values)
    C_array = np.array(real_data_all)  # shape (N_cfg, T)

    N = C_array.shape[0]  # files num
    T = C_array.shape[1]  # time-extent, 96 in your case

    # folded length: include t=0 .. t=T//2 (inclusive)
    n_half = T // 2 + 1  # for T=96, n_half=49

    # 对每个配置做对称化：folded_config[t] = 0.5*(C[t] + C[(T-t)%T])
    flip_data = np.zeros((N, n_half))
    for i_cfg in range(N):
        cfg = C_array[i_cfg]
        for t in range(n_half):
            tp = (T - t) % T
            flip_data[i_cfg, t] = 0.5 * (cfg[t] + cfg[tp])

    t = np.arange(n_half)
    print(f"{N} 组数据，{len(t)} 个时间点 (折叠后，T={T})")

    return flip_data


def load_zeromom_data(path):
    """
    load zero-momentum correlator data from .dat files
    Return:
        zeromom_data: folded & averaged data per config
        t: time data (0 .. T//2)
        N: number of correlator files
    """
    mumus = [0, 1, 2]
    data = np.zeros((15, 49))
    for mumu in mumus:
        data += load_data(path, mumu)
    print(f"load all {len(mumus)} mumu data.")
    return data / 3


if __name__ == "__main__":
    # flip_data, t, N = load_data(path="../data/raw/mass", mumu=0)

    # print(flip_data.shape)
    data = load_zeromom_data(path="./data/raw/mass")
