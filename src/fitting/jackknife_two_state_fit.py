import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, fit_report

from src.models.residuals import make_uncorrelated_residual
from src.fitting.two_state_fit import two_state_fit
from src.utils.io import load_data
from src.models.model_functions import two_cosh_func
from src.utils.stats import jackknife_resample


def jackknife_fit(t_fit,T,jk_samples,sigma,residual,scale_factor,print_report=False):
    """
    uncorrelated 2 state jackknife fit
    """

    # 对每个jackknife sample拟合
    jk_params_A0 = []
    jk_params_m0 = []
    jk_params_A1 = []
    jk_params_m1 = []
    failed_fits = 0

    N_blocks = len(jk_samples)

    for i in range(N_blocks):
        result = two_state_fit(t_fit=t_fit, data_fit=jk_samples[i,:], err_fit=sigma,
                               T=T,residual=residual,scale_factor=scale_factor)

        if result.success:
            jk_params_A0.append(result.params['A0'].value)
            jk_params_A1.append(result.params['A1'].value)
            jk_params_m0.append(result.params['m0'].value)
            jk_params_m1.append(result.params['m1'].value)
        else:
            failed_fits += 1

    if len(jk_params_A0) == 0:
        raise RuntimeError("All jackknife fit failed")

    # 进行估计
    jk_params_A0 = np.array(jk_params_A0) / scale_factor  # 重新放缩
    jk_params_A1 = np.array(jk_params_A1) / scale_factor
    jk_params_m0 = np.array(jk_params_m0)
    jk_params_m1 = np.array(jk_params_m1)

    # 计算mean和err
    A0_mean = np.mean(jk_params_A0)
    A0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_A0, ddof=0))

    m0_mean = np.mean(jk_params_m0)
    m0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_m0, ddof=0))

    A1_mean = np.mean(jk_params_A1)
    A1_std = np.sqrt((N_blocks - 1) * np.var(jk_params_A1, ddof=0))

    m1_mean = np.mean(jk_params_m1)
    m1_std = np.sqrt((N_blocks - 1) * np.var(jk_params_m1, ddof=0))

    # 输出部分
    class JackknifeResult:
        def __init__(self, A0_mean, A0_std, A1_mean, A1_std, m0_mean, m0_std, m1_mean, m1_std, success_rate) -> None:
            self.params = {
                'A0': type("", (), {"value": A0_mean, 'stderr': A0_std})(),
                'A1': type("", (), {"value": A1_mean, 'stderr': A1_std})(),
                'm0': type("", (), {"value": m0_mean, 'stderr': m0_std})(),
                'm1': type("", (), {"value": m1_mean, 'stderr': m1_std})(),
            }
            self.success_rate = success_rate
            self.scale_factor = scale_factor
            self.aic = result.aic
            self.bic = result.bic
            self.redchi = result.redchi

    if print_report:
        print("=" * 50)
        print("Jackknife two state fit result")
        # print(
        #     f"success rate: {len(jk_params_A0)}/{N_blocks} ({100*len(jk_params_A0) / N_blocks)})"
        # # )
        print(f"A0: {A0_mean}  ± {A0_std} ")
        print(f"A1: {A1_mean}  ± {A1_std} ")
        print(f"m0: {m0_mean}  ± {m0_std}")
        print(f"m1: {m1_mean}  ± {m1_std}")

    return JackknifeResult(
        A0_mean, A0_std, A1_mean, A1_std, m0_mean, m0_std, m1_mean, m1_std, len(jk_params_A0) / N_blocks
    )


def main(show_plot=True):
    data, t, _ = load_data(path="../data/raw/mass/",mumu=0)
    T = 96

    data = data*1e10
    residual = make_uncorrelated_residual(two_cosh_func)
    mask = (t >= 5) & (t <= 16)
    data_masked = data[:, mask]
    t_masked = t[mask]

    jk_samples,jk_mean,jk_err = jackknife_resample(data=data_masked,return_samples=True)
    jackknife_fit(t_fit=t_masked,T=T,jk_samples=jk_samples,sigma=jk_err,residual=residual,scale_factor=1e10,print_report=True)



if __name__ == "__main__":
    main()
