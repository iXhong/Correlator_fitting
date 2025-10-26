import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from load_data import data_load
import warnings

# 复用你原来的 two_cosh_func / residual / two_state_fit
def two_cosh_func(params, t, T):
    # function to fit
    A0 = params['A0']
    A1 = params['A1']
    m0 = params['m0']
    m1 = params['m1']
    return A0 * np.cosh(m0 * (t - T / 2)) + A1 * np.cosh(m1 * (t - T / 2))


def double_cosh_residual(params, t_fit, data_fit, err_fit, T):
    model = two_cosh_func(params, t_fit, T)
    return (data_fit - model) / err_fit


def two_state_fit(t_fit, data_fit, err_fit, T, scale_factor):
    params_double = Parameters()
    params_double.add('A0', value=1e-15 * scale_factor, min=1e-16 * scale_factor)
    params_double.add('A1', value=1e-17 * scale_factor, min=0, max=1e-7 * scale_factor)
    params_double.add('m0', value=0.60, min=0.6, max=0.8)
    params_double.add('m1', value=1.0, min=0.7, max=5.0)

    # wrap minimize call in try/except to avoid one bad sample killing everything
    try:
        result = minimize(
            double_cosh_residual,
            params_double,
            method='least_squares',
            kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit, "T": T}
        )
    except Exception as e:
        # convert exceptions to a fake result-like object with success False
        class _Fail:
            success = False
        result = _Fail()
    return result


def bootstrap_two_state_fit(t_min: int, t_max: int, t, data, T: int,
                            n_resamples: int = 500,
                            random_seed: int = 1234,
                            print_report: bool = True):
    """
    Bootstrap uncorrelated two-state fit (结构参照 jackknife_fit)
    Args:
        t_min, t_max: 拟合区间
        t: 时间点数组
        data: shape (N_cfg, N_t)
        T: Euclidean 时间 length
        n_resamples: bootstrap 重抽样次数
        random_seed: RNG seed
        print_report: 是否打印结果
    Returns:
        BootstrapResult: 包含 params (A0, A1, m0, m1) 和相应 stderr，success_rate，scale_factor，及参数样本列表
    """

    rng = np.random.default_rng(random_seed)
    N_block = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]

    # 与 jackknife 保持一致的 scale_factor
    scale_factor = 1e10
    data_scaled = data * scale_factor

    # 构造 bootstrap samples (对配置有放回重抽样) 并取均值
    bs_means = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, N_block, N_block)  # 有放回采样
        bs_subdata = data_scaled[chosen][:, fit_mask]
        bs_mean = np.mean(bs_subdata, axis=0)
        bs_means.append(bs_mean)
    bs_means = np.array(bs_means)  # shape (n_resamples, n_tfit)

    # 全局误差（用于 WLS 的 err_fit）：采用与你单态实现相同的修正
    # sigma_global = sqrt(N/(N-1)) * std(bs_means, ddof=1)
    # ddof=1 already gives unbiased estimator; 乘 sqrt(N/(N-1)) 是某些实现的修正，保持你原来做法
    with np.errstate(invalid='ignore'):
        sigma_global = np.sqrt(N_block / (N_block - 1)) * np.std(bs_means, axis=0, ddof=1)
        # 防止有0导致拟合除以0
        sigma_global[sigma_global == 0] = np.min(sigma_global[sigma_global > 0]) if np.any(sigma_global > 0) else 1.0

    # 对每个 bootstrap sample 做 two-state 拟合
    bs_params_A0 = []
    bs_params_A1 = []
    bs_params_m0 = []
    bs_params_m1 = []
    failed_fits = 0

    for i in range(n_resamples):
        bs_data_fit = bs_means[i]

        # 直接复用 two_state_fit
        result = two_state_fit(t_fit=t_fit, data_fit=bs_data_fit, err_fit=sigma_global, T=T, scale_factor=scale_factor)

        if getattr(result, "success", False):
            try:
                bs_params_A0.append(result.params['A0'].value)
                bs_params_A1.append(result.params['A1'].value)
                bs_params_m0.append(result.params['m0'].value)
                bs_params_m1.append(result.params['m1'].value)
            except Exception:
                failed_fits += 1
        else:
            failed_fits += 1

    if len(bs_params_A0) == 0:
        raise RuntimeError("所有 bootstrap 两态拟合都失败了")

    # 转为 numpy arrays 并放缩回原量级（A 参数除以 scale_factor）
    bs_params_A0 = np.array(bs_params_A0) / scale_factor
    bs_params_A1 = np.array(bs_params_A1) / scale_factor
    bs_params_m0 = np.array(bs_params_m0)
    bs_params_m1 = np.array(bs_params_m1)

    # 使用 median 作为点估计，std(ddof=1) 作为误差（与你单态实现一致）
    A0_med = np.median(bs_params_A0)
    A0_std = np.std(bs_params_A0, ddof=1)

    A1_med = np.median(bs_params_A1)
    A1_std = np.std(bs_params_A1, ddof=1)

    m0_med = np.median(bs_params_m0)
    m0_std = np.std(bs_params_m0, ddof=1)

    m1_med = np.median(bs_params_m1)
    m1_std = np.std(bs_params_m1, ddof=1)

    # 平均 redchi / aic 在 two-state 中可能不可用（最小化器未返回），所以不强制收集
    success_rate = len(bs_params_A0) / n_resamples

    class BootstrapResult:
        def __init__(self, A0_med, A0_std, A1_med, A1_std,
                     m0_med, m0_std, m1_med, m1_std,
                     success_rate, scale_factor,
                     bs_params_A0, bs_params_A1, bs_params_m0, bs_params_m1):
            self.params = {
                'A0': type("", (), {"value": A0_med, "stderr": A0_std})(),
                'A1': type("", (), {"value": A1_med, "stderr": A1_std})(),
                'm0': type("", (), {"value": m0_med, "stderr": m0_std})(),
                'm1': type("", (), {"value": m1_med, "stderr": m1_std})(),
            }
            self.success = True
            self.success_rate = success_rate
            self.scale_factor = scale_factor
            # 保存样本以便后续分析（直方图等）
            self.bs_params_A0 = bs_params_A0
            self.bs_params_A1 = bs_params_A1
            self.bs_params_m0 = bs_params_m0
            self.bs_params_m1 = bs_params_m1

    if print_report:
        print("=" * 50)
        print("Bootstrap 两态拟合结果:")
        print(f"成功率: {len(bs_params_A0)}/{n_resamples} ({100 * success_rate:.1f}%)")
        print(f"A0: {A0_med:.6e} ± {A0_std:.6e}")
        print(f"A1: {A1_med:.6e} ± {A1_std:.6e}")
        print(f"m0: {m0_med:.6f} ± {m0_std:.6f}")
        print(f"m1: {m1_med:.6f} ± {m1_std:.6f}")

    return BootstrapResult(
        A0_med, A0_std, A1_med, A1_std,
        m0_med, m0_std, m1_med, m1_std,
        success_rate, scale_factor,
        bs_params_A0, bs_params_A1, bs_params_m0, bs_params_m1
    )


def plot_bs_result(out, mean_corr, t, T):
    A0 = out.params['A0'].value
    A1 = out.params['A1'].value
    m0 = out.params['m0'].value
    m1 = out.params['m1'].value

    def two_state_cosh(t, A0, m0, A1, m1, T):
        return (A0 * np.cosh(m0 * (t - T / 2)) +
                A1 * np.cosh(m1 * (t - T / 2)))

    y = two_state_cosh(t, A0, m0, A1, m1, T)

    plt.figure(figsize=(10, 6))
    plt.plot(t, np.log(np.abs(mean_corr)), 'o', label='mean data')
    plt.plot(t, np.log(y), '-', label='bootstrap fit')
    plt.xlabel('t')
    plt.ylabel('log(|C(t)|)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main(show_plot=True):
    data, t, _ = data_load()
    T = 96

    out = bootstrap_two_state_fit(t_min=5, t_max=16, t=t, data=data, T=T, n_resamples=100, print_report=True)

    if show_plot:
        mean_corr = np.mean(data, axis=0)
        plot_bs_result(out, mean_corr, t, T)

    return out


if __name__ == "__main__":
    # 举例运行
    out = main(show_plot=True)


