"""
Description: Jackknife fitting method for one-state model.
@author: GeorgeLiu
@since: 2025.11
"""

import numpy as np
from src.fitting.base_fit import FitResult,FitMethod
from lmfit import Parameters,minimize


class JackknifeOneStateFit(FitMethod):

    def _init_params(self, data_fit) -> Parameters:
        # initialize fit parameters
        params = Parameters()
        params.add("A0", value=data_fit[0], min=0)
        params.add("m0", value=0.5, min=0)
        return params

    def _run_fit(self,t_fit,T,jk_samples,sigma,residual,print_report=True) -> FitResult:
        # main jackknife fit routine for one-state model
        N_blocks = jk_samples.shape[0]

        jk_params_A0 = []
        jk_params_m0 = []
        jk_redchi = []
        jk_aicc = []

        failed_fits = 0

        for i_block in range(N_blocks):
            jk_data_fit = jk_samples[i_block]

            params = self._init_params(jk_data_fit)

            try:
                result = minimize(
                    residual,
                    params,
                    method="least_squares",
                    kws={
                        "t_fit": t_fit,
                        "data_fit": jk_data_fit,
                        "err_fit": sigma,
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
            raise RuntimeError("all jackknife fits failed.")

        # 计算参数的 jackknife 误差
        # p0_std = sqrt(N-1)*std(p0)
        jk_params_A0 = np.array(jk_params_A0)
        jk_params_m0 = np.array(jk_params_m0)

        A0_mean = np.mean(jk_params_A0)
        A0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_A0, ddof=0))

        m0_mean = np.mean(jk_params_m0)
        m0_std = np.sqrt((N_blocks - 1) * np.var(jk_params_m0, ddof=0))

        success_rate = (N_blocks - failed_fits) / N_blocks

        # construct FitResult
        out = FitResult.create1param(
            A0=A0_mean,
            A0_err=A0_std,
            m0=m0_mean,
            m0_err=m0_std,
            redchi=np.mean(jk_redchi),
            aic=np.mean(jk_aicc),
            success_rate=success_rate,
            method="Jackknife Fit",
        )

        # print report
        if print_report:
            print("=" * 20)
            print("Jackknife拟合结果:")
            print(f"A0 = {A0_mean} ± {A0_std}")
            print(f"m0 = {m0_mean:.6f} ± {m0_std}")
            print(f"平均 reduced chi-squared: {out.redchi:.3f}")
            print(f"平均 AICc: {out.aic:.3f}")
            print(
                f"成功率: {N_blocks - failed_fits}/{N_blocks} ({100 * success_rate:.1f}%)"
            )
            print("=" * 20)

        return out

