"""
resampling methods
@author GeorgeLiu
@since 2025.10
"""
import numpy as np


def jackknife_resample(data, return_samples=False):
    """
    Jackknife resampling function
    :param data: 输入数据, shape (N_blocks, n_t_fit)
    其中N_blocks是jackknife块的数量,n_t_fit是拟合点的数量
    例如data可以是形状为(100,12)的二维数组,
    :param return_samples:
    :return:
    jk_samples : array, shape (N_blocks, n_t_fit), optional
        当return_samples=True时返回所有jackknife样本
    jk_mean : array, shape (n_t_fit,)
        jackknife样本均值
    jk_err : array, shape (n_t_fit,)
        jackknife标准误差
    计算方法参考:
    https://en.wikipedia.org/wiki/Jackknife_resampling
    """
    N_blocks = data.shape[0]
    jk_samples = []

    for i in range(N_blocks):
        jk_indices = np.concatenate([np.arange(i), np.arange(i + 1, N_blocks)])
        jk_sample = np.mean(data[jk_indices], axis=0)
        jk_samples.append(jk_sample)
    jk_samples = np.array(
        jk_samples
    )  # this is jackknife samples, shape (N_blocks, n_t_fit)

    # 计算全局误差
    jk_mean = np.mean(jk_samples, axis=0)
    # diffs = jk_samples - jk_mean
    # jackknife 方差公式 (N-1)/N * sum (diff^2)等价于sqrt(N-1)*np.std(jk_samples)
    # jk_err = np.sqrt((N_blocks - 1) / N_blocks * np.sum(diffs**2, axis=0))
    # 你会发现jk_err和np.std(data)差别挺大,为什么呢?因为这二者不是一个东西,jk_err是统计估计的标准误
    # np.std是样本的标准差,二者的关系是err = std/sqrt(N)
    # 事实上我们这里只是使用了均值估计,我们也可以不使用均值,使用中位数,方差等等,这时jackknife估计就要相应的改变,但是误差是不变的.
    jk_err = np.sqrt(N_blocks - 1) * np.std(jk_samples, axis=0, ddof=0)

    if return_samples:
        return jk_samples, jk_mean, jk_err
    else:
        return jk_mean, jk_err


def bootstrap_resample(data, n_resamples=1000, return_samples=False, random_state=42):
    """
    Bootstrap重采样函数

    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        输入数据
    n_resamples : int, default=1000
        bootstrap重采样次数
    return_samples : bool, default=False
        是否返回所有bootstrap样本
    random_state : int, default=42
        随机种子

    Returns:
    --------
    bs_samples : array, shape (n_resamples, n_features), optional
        当return_samples=True时返回所有bootstrap样本
    bs_mean : array, shape (n_features,)
        bootstrap样本均值
    bs_err : array, shape (n_features,)
        bootstrap标准误差
    """
    N_blocks = data.shape[0]
    rng = np.random.default_rng(random_state)

    # 更高效的向量化实现
    indices = rng.integers(0, N_blocks, size=(n_resamples, N_blocks))
    bs_samples = np.mean(data[indices], axis=1)

    bs_mean = np.mean(bs_samples, axis=0)
    bs_err = np.sqrt(N_blocks / (N_blocks - 1)) * np.std(bs_samples, axis=0, ddof=1)

    if return_samples:
        return bs_samples, bs_mean, bs_err
    else:
        return bs_mean, bs_err


if __name__ == "__main__":
    # data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    np.random.seed(42)
    data = np.random.normal(0, 1, 2000)
    jackknife_resample(data)
    bootstrap_resample(data)
    print(1 / np.sqrt(2000))
    # samples,mean,err = jackknife_resample(data,return_samples=True)
    # print(mean)
    # print(np.sqrt(9)*np.std(data))
    # print(err)
