import numpy as np
import glob
import os


def load_one_config(file_path, p2, nt):
    """Load data from one configuration file."""
    data = np.loadtxt(file_path, comments="#")
    p2_block = data[data[:, 0] == p2]
    times = p2_block[:, 1]
    correlators = p2_block[:, 2][times == nt]
    print(correlators)


def construct_time_slice(indir, p2, nt):
    """Construct time slice data for all configurations."""
    file_list = glob.glob(os.path.join(indir, "*.txt"))
    all_correlators = []
    for file_path in file_list:
        data = np.loadtxt(file_path, comments="#")
        p2_block = data[data[:, 0] == p2]
        times = p2_block[:, 1]
        correlators = p2_block[:, 2][times == nt]
        all_correlators.append(correlators[0])
    all_correlators = np.array(all_correlators)
    # print(all_correlators)
    return all_correlators


def autocorr_time(x, c=4.0, t_max=None):
    """
    估计一维时间序列 x 的 integrated autocorrelation time τ_int

    参数
    ----
    x : array-like, shape (N,)
        马尔可夫链上的观测值序列
    c : float
        窗口因子，通常 4 ~ 6 之间。窗口通过条件 t > c * tau_int(t) 自动确定
    t_max : int or None
        计算自相关函数的最大滞后时间。如果为 None，则默认 N//2

    返回
    ----
    tau_int : float
        估计的积分自相关时间
    t_arr : ndarray
        自相关函数的滞后时间数组 (0..t_max)
    rho : ndarray
        标准化自相关函数 ρ(t)，长度 = t_max+1
    """
    x = np.asarray(x, dtype=float)
    N = x.size

    if t_max is None:
        t_max = N // 2  # 保守一点，最多取一半长度

    # 去平均
    x = x - x.mean()

    # 通过 np.correlate 计算自协方差（非归一化）
    # mode="full" → 长度 2N-1，中点对应滞后 0
    corr = np.correlate(x, x, mode="full")
    corr = corr[N - 1 : N - 1 + t_max + 1]

    # 归一化为 Γ(t)，注意分母 (N - t)
    t_arr = np.arange(t_max + 1)
    gamma = corr / (N - t_arr)

    # 标准化自相关函数 ρ(t)
    gamma0 = gamma[0]
    rho = gamma / gamma0

    # 自适应窗口估计 τ_int
    tau_int = 0.5  # 从 1/2 开始
    for t in range(1, t_max + 1):
        tau_int += rho[t]
        # 窗口条件：t > c * tau_int(t)
        if t > c * tau_int:
            break

    return tau_int, t_arr, rho


# load_one_config(
#     "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/p2_sorted/TwoPt_ss_z2_conf210_mom333_p2.txt",
#     p2=0,
#     nt=0,
# )

if __name__ == "__main__":
    corr = construct_time_slice(
        "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/p2_sorted/",
        p2=0,
        nt=15,
    )

    tau_int, t_arr, rho = autocorr_time(corr, c=4.0)
    print("Estimated integrated autocorrelation time:", tau_int)
