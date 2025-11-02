import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from load_data import data_load
import warnings


def fit_function(params, t, T):
    A0 = params['A0']
    A1 = params['A1']
    m0 = params['m0']
    m1 = params['m1']
    return A0 * np.cosh(m0 * (t - T / 2)) + A1 * np.cosh(m1 * (t - T / 2))


def correlated_residual(params, t_fit, data_fit, yerr_fit, corr_matrix,T):

    x = t_fit
    y = data_fit
    sigma = yerr_fit
    # model = model_func(params, x)
    model = fit_function(params,x,T)
    delta_norm = (y - model) / sigma
    
    # Compute inverse of correlation matrix
    inv_corr = np.linalg.inv(corr_matrix)
    
    # Cholesky decomposition of inv_corr: L such that L @ L.T = inv_corr
    L = np.linalg.cholesky(inv_corr)
    
    # Residuals: L.T @ delta_norm so that res.T @ res = delta_norm.T @ inv_corr @ delta_norm
    res = np.dot(L.T, delta_norm)
    
    return res


def two_state_fit_uncorr(t_fit, data_fit, err_fit, T, scale_factor):
    """保留你原来的 two_state_fit 用于 uncorrelated case（不改动）"""
    params_double = Parameters()
    params_double.add('A0', value=1e-15 * scale_factor, min=1e-16 * scale_factor)
    params_double.add('A1', value=1e-17 * scale_factor, min=0, max=1e-7 * scale_factor)
    params_double.add('m0', value=0.60, min=0.6, max=0.8)
    params_double.add('m1', value=1.0, min=0.7, max=5.0)

    try:
        result = minimize(
            double_cosh_residual_uncorr,
            params_double,
            method='least_squares',
            kws={"t_fit": t_fit, "data_fit": data_fit, "err_fit": err_fit, "T": T}
        )
    except Exception as e:
        class _Fail:
            success = False
        result = _Fail()
    return result


def two_state_fit_correlated(t_fit, data_fit, L_lower, T, scale_factor):
    """
    Correlated two-state fit: use L_lower (cholesky factor of cov_mean) to whiten residuals.
    L_lower shape: (n_tfit, n_tfit) lower-triangular such that cov_mean = L_lower @ L_lower.T
    """
    params_double = Parameters()
    params_double.add('A0', value=1e-15 * scale_factor, min=1e-16 * scale_factor)
    params_double.add('A1', value=1e-17 * scale_factor, min=0, max=1e-7 * scale_factor)
    params_double.add('m0', value=0.60, min=0.6, max=0.8)
    params_double.add('m1', value=1.0, min=0.7, max=5.0)

    try:
        result = minimize(
            double_cosh_residual_correlated,
            params_double,
            method='least_squares',
            kws={"t_fit": t_fit, "data_fit": data_fit, "L_lower": L_lower, "T": T}
        )
    except Exception as e:
        # convert exceptions to fake result
        class _Fail:
            success = False
        result = _Fail()
    return result


def bootstrap_two_state_fit_correlated(t_min: int, t_max: int, t, data, T: int,
                                       n_resamples: int = 500,
                                       random_seed: int = 1234,
                                       reg_cov: float = 1e-8,
                                       print_report: bool = True):
    """
    Bootstrap correlated two-state fit（基于你的 uncorrelated 版本做改造）
    For each bootstrap sample:
      - sample configurations with replacement -> bs_subdata (N_block, n_tfit)
      - compute bs_mean (n_tfit,)
      - compute covariance of configurations: cov_configs (n_tfit, n_tfit)
      - covariance of the mean: cov_mean = cov_configs / N_block
      - regularize cov_mean and compute cholesky L_lower s.t. cov_mean = L_lower @ L_lower.T
      - fit by minimizing || L_lower^{-1} (bs_mean - model) ||^2  (i.e. correlated chi^2)
    Args:
        reg_cov: relative regularization amplitude; actual reg added = reg_cov * trace(cov_mean)/n_tfit
    Returns:
        BootstrapResult object similar to uncorrelated version
    """
    rng = np.random.default_rng(random_seed)
    N_block = data.shape[0]
    fit_mask = (t >= t_min) & (t <= t_max)
    t_fit = t[fit_mask]
    n_tfit = t_fit.size

    # scale factor 与 jackknife 保持一致
    scale_factor = 1e10
    data_scaled = data * scale_factor

    bs_params_A0 = []
    bs_params_A1 = []
    bs_params_m0 = []
    bs_params_m1 = []
    failed_fits = 0

    # 为了避免过多奇异矩阵导致失败，我们在无法 cholesky 时做逐步增大正则化的策略
    def make_L_cholesky(cov, base_reg=reg_cov):
        """
        Try Cholesky; if fails, add reg * I with reg increased up to some limit.
        Returns L_lower (lower-triangular) or None if totally fails.
        """
        trace = np.trace(cov)
        # if trace == 0, put small absolute floor
        if trace == 0:
            trace = 1.0
        for k in range(0, 8):
            reg = base_reg * (10.0**k) * (trace / n_tfit)
            cov_reg = cov + reg * np.eye(n_tfit)
            try:
                L = np.linalg.cholesky(cov_reg)  # lower-triangular
                return L
            except np.linalg.LinAlgError:
                continue
        # last resort: symmetrize and use eig to build sqrt if positive
        try:
            vals, vecs = np.linalg.eigh(cov)
            # clip eigenvalues to small positive
            vals_clipped = np.clip(vals, a_min=1e-16 * (trace / n_tfit), a_max=None)
            cov_ps = (vecs * vals_clipped) @ vecs.T
            L = np.linalg.cholesky(cov_ps + 1e-16 * np.eye(n_tfit))
            return L
        except Exception:
            return None

    for i in range(n_resamples):
        # sample configurations with replacement
        chosen = rng.integers(0, N_block, N_block)
        bs_subdata = data_scaled[chosen][:, fit_mask]  # shape (N_block, n_tfit)
        bs_mean = np.mean(bs_subdata, axis=0)  # shape (n_tfit,)

        # compute configuration-level covariance (rows = samples, cols = t)
        # np.cov rowvar=False -> shape (n_tfit, n_tfit)
        if N_block > 1:
            cov_configs = np.cov(bs_subdata, rowvar=False, ddof=1)  # cov over configs
        else:
            cov_configs = np.zeros((n_tfit, n_tfit))

        # covariance of the mean
        cov_mean = cov_configs / float(N_block)

        # regularize & attempt cholesky
        L = make_L_cholesky(cov_mean, base_reg=reg_cov)

        if L is None:
            # 严重退化：退回到 uncorrelated 使用对角 sigma（避免中断整个bootstrap）
            # 采用各个时间点的 sqrt(diag(cov_mean)) 作为 err
            diag_var = np.diag(cov_mean).copy()
            # if zeros set to small positive
            diag_var[diag_var <= 0] = np.min(diag_var[diag_var > 0]) if np.any(diag_var > 0) else 1.0
            err_fit = np.sqrt(diag_var)
            # 为了和 uncorrelated 的 residual 接口匹配，我们把 err_fit 中的 0 值替换为 1
            err_fit[err_fit == 0] = 1.0
            result = two_state_fit_uncorr(t_fit=t_fit, data_fit=bs_mean, err_fit=err_fit, T=T, scale_factor=scale_factor)
        else:
            # use correlated fitter
            result = two_state_fit_correlated(t_fit=t_fit, data_fit=bs_mean, L_lower=L, T=T, scale_factor=scale_factor)

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
        raise RuntimeError("所有 correlated bootstrap 两态拟合都失败了")

    # 转为 numpy arrays 并放缩回原量级（A 参数除以 scale_factor）
    bs_params_A0 = np.array(bs_params_A0) / scale_factor
    bs_params_A1 = np.array(bs_params_A1) / scale_factor
    bs_params_m0 = np.array(bs_params_m0)
    bs_params_m1 = np.array(bs_params_m1)

    # 使用 median 作为点估计，std(ddof=1) 作为误差
    A0_med = np.median(bs_params_A0)
    A0_std = np.std(bs_params_A0, ddof=1)

    A1_med = np.median(bs_params_A1)
    A1_std = np.std(bs_params_A1, ddof=1)

    m0_med = np.median(bs_params_m0)
    m0_std = np.std(bs_params_m0, ddof=1)

    m1_med = np.median(bs_params_m1)
    m1_std = np.std(bs_params_m1, ddof=1)

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
        print("Bootstrap 两态拟合 (CORRELATED) 结果:")
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


# Plotting 函数与 main 保持一致，显示拟合曲线（用返回的 median 值）
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
    plt.plot(t, np.log(y), '-', label='bootstrap correlated fit')
    plt.xlabel('t')
    plt.ylabel('log(|C(t)|)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main(show_plot=True):
    data, t, _ = data_load()
    T = 96

    out = bootstrap_two_state_fit_correlated(t_min=5, t_max=16, t=t, data=data, T=T, n_resamples=100, print_report=True)

    if show_plot:
        mean_corr = np.mean(data, axis=0)
        plot_bs_result(out, mean_corr, t, T)

    return out


if __name__ == "__main__":
    out = main(show_plot=True)
