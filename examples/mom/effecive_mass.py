import numpy as np
import glob
import os
import matplotlib.pyplot as plt


# def effective_mass(t, C_t_mean):
#     """计算有效质量"""
#     m_eff = np.log(C_t_mean[:-1] / C_t_mean[1:])
#     return m_eff


import numpy as np


def eff_mass_cosh(C):
    """
    计算 cosh 形式的有效质量 m_eff(t)
    公式:
        m_eff(t) = arcosh( (C[t+1] + C[t-1]) / (2*C[t]) )

    参数
    ----
    C : np.ndarray
        correlator，形状为 (T,) 或 (Nbs, T)

    返回
    ----
    t_eff : ndarray
        有效质量定义在 t = 1..T-2
    m_eff : ndarray
        有效质量数组
        - 若输入 shape = (T,)  → m_eff shape = (T-2,)
        - 若输入 shape = (Nbs, T) → (Nbs, T-2)
    """
    C = np.asarray(C)
    T = C.shape[-1]

    # 中心三点公式（支持广播）
    num = C[..., 2:] + C[..., :-2]  # C(t+1) + C(t-1)
    den = 2.0 * C[..., 1:-1]  # 2*C(t)

    ratio = num / den

    # 数值保护：ratio 必须 >= 1，否则 arcosh 无定义
    ratio = np.maximum(ratio, 1.0)

    m_eff = np.arccosh(ratio)
    t_eff = np.arange(1, T - 1)

    return t_eff, m_eff


# def get_effective_mass(file_path):

#     samples = np.load(file_path)
#     print(samples.shape)
#     m_eff_list = []

#     for i in range(samples.shape[0]):
#         C_t = samples[i, :]
#         m_eff = effective_mass(np.arange(len(C_t)), C_t)
#         m_eff_list.append(m_eff)
#     m_eff_array = np.array(m_eff_list)
#     m_eff_mean = np.mean(m_eff_array, axis=0)
#     m_eff_err = np.std(m_eff_array, axis=0, ddof=1)
#     return m_eff_mean, m_eff_err


if __name__ == "__main__":
    # t, C_t_mean, C_t_err = data_load(
    #     "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_folded/bs_folded_p2_0_mean_err.dat"
    # )
    file_path = "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_samples/phi_p2_0_bs.npy"
    # m_eff_mean, m_eff_err = get_effective_mass(file_path)
    # print(m_eff_mean.shape)
    # print(m_eff_mean)

    C_bs = np.load(file_path)  # shape (Nbs, T)
    t_eff, m_eff_bs = eff_mass_cosh(C_bs)

    m_eff_mean = m_eff_bs.mean(axis=0)
    m_eff_err = m_eff_bs.std(axis=0, ddof=1)

    plt.figure()
    plt.errorbar(
        np.arange(len(m_eff_mean)),
        m_eff_mean,
        yerr=m_eff_err,
        fmt="o",
    )
    # plt.ylim(0.5, 0.9)
    # plt.xlim(0, 30)
    plt.xlabel(r"$\tau/a$")
    plt.ylabel(r"$am_{eff}$")
    plt.grid()
    plt.show()
