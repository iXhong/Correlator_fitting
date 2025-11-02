"""
jackknife fit
@author George Liu
@since 2025.10
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from lmfit import Parameters, minimize,fit_report
from src.models.model_functions import two_cosh_func, one_cosh_func
from src.utils.io import load_data
from src.models.residuals import make_uncorrelated_residual
from src.utils.stats import jackknife_resample


def jackknife_fit(t_fit,T,jk_samples,sigma,residual,print_report=True):
    """
    uncorrelated 1 state jackknife fit
    Args:
        print_report:print report or not
    Returns:
        JackknifeResult: a class like params in lmfit
    Examples:
        m0 = result.params['m0'].value
    """
    N_blocks = jk_samples.shape[0]

    # 对每个 jackknife sample 做拟合
    jk_params_A0 = []
    jk_params_m0 = []
    jk_redchi = []
    jk_aicc = []
    failed_fits = 0

    for i in range(N_blocks):
        jk_data_fit = jk_samples[i]

        params = Parameters()
        params.add("A0", value=jk_data_fit[0], min=0)
        params.add("m0", value=0.5, min=0)

        try:
            result = minimize(
                residual,
                params,
                method="least_squares",
                kws={
                    "t_fit": t_fit,
                    "data_fit": jk_data_fit,
                    "er r_fit": sigma,
                    "T": T,
                },
            )
            if result.success:
                jk_params_A0.append(result.params["A0"].value)
                jk_params_m0.append(result.params["m0"].value)
                jk_redchi.append(result.redchi)
                jk_aicc.append(result.aic)
            else:
                failed_fits += 1


        except Exception:
            failed_fits += 1

    if len(jk_params_A0) == 0:
        raise RuntimeError("所有jackknife拟合都失败了")

    # 计算参数的 jackknife 误差
    # p0_std = sqrt(N-1)*std(p0)
    jk_params_A0 = np.array(jk_params_A0)
    jk_params_m0 = np.array(jk_params_m0)

    A0_mean = np.mean(jk_params_A0)
    A0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_A0, ddof=0))

    m0_mean = np.mean(jk_params_m0)
    m0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_m0, ddof=0))

    redchi = np.mean(jk_redchi)
    aicc = np.mean(jk_aicc)

    # 输出
    class JackknifeResult:
        def __init__(self, A0_mean, A0_std, m0_mean, m0_std, redchi,aicc,success_rate):
            self.params = {
                "A0": type("", (), {"value": A0_mean, "stderr": A0_std})(),
                "m0": type("", (), {"value": m0_mean, "stderr": m0_std})(),
            }
            self.success = True
            self.redchi = redchi
            self.aicc = aicc
            self.success_rate = success_rate

    if print_report:
        print("=" * 50)
        print("Jackknife拟合结果:")
        print(
            f"成功率: {len(jk_params_A0)}/{N_blocks} ({100 * len(jk_params_A0) / N_blocks:.1f}%)"
        )
        print(f"A0: {A0_mean} ± {A0_std}")
        print(f"m0: {m0_mean:.6f} ± {m0_std:.6f}")

    return JackknifeResult(
        A0_mean, A0_std, m0_mean, m0_std, redchi, aicc, len(jk_params_A0) / N_blocks
    )


if __name__ == "__main__":
    data,t,N = load_data(path="../data/raw/mass",mumu=0)
    residual = make_uncorrelated_residual(one_cosh_func)
    mask = (t >= 5) & (t <= 16)
    data_masked = data[:,mask]
    t_masked = t[mask]

    jk_samples,jk_mean,jk_err = jackknife_resample(data=data_masked,return_samples=True)

    jackknife_fit(t_fit=t_masked,T=96,jk_samples=jk_samples,sigma=jk_err,residual=residual,print_report=True)

