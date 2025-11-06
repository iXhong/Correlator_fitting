"""
Description: Direct fitting method for one-state model.
@author: GeorgeLiu
@since: 2025.11
"""

import numpy as np
from src.fitting.base_fit import FitResult, FitMethod
from lmfit import Parameters, minimize


class DirectFit(FitMethod):

    def _init_params(self, data_fit) -> Parameters:
        # initialize fit parameters
        params = Parameters()
        params.add("A0", value=data_fit[0], min=0)
        params.add("m0", value=0.5, min=0)
        return params

    def _run_fit(
        self, t_fit, T, data_fit, sigma, residual, print_report=True
    ) -> FitResult:
        """
        main direct fit routine for one-state model
        :param t_fit:
        :param T:
        :param data_fit:
        :param sigma: global error of data_fit as sigma
        :param residual: redisual function
        :param print_report: bool, print fit report or not
        :return: FitResult
        """
        params = self._init_params(data_fit)

        result = minimize(
            residual,
            params,
            method="least_squares",
            kws={
                "t_fit": t_fit,
                "data_fit": data_fit,
                "err_fit": sigma,
                "T": T,
            },
        )

        if result.success:
            A0 = result.params["A0"].value
            A0_err = result.params["A0"].stderr
            m0 = result.params["m0"].value
            m0_err = result.params["m0"].stderr
            redchi = result.redchi
            aic = result.aic
            success_rate = 1.0

        else:
            A0 = A0_err = m0 = m0_err = redchi = aic = success_rate = np.nan

        if print_report:
            print("=" * 20)
            print("Direct Fit Report:")
            print(f"A0 = {A0} ± {A0_err}")
            print(f"m0 = {m0} ± {m0_err}")
            print(f"Reduced chi-squared = {redchi}")
            print(f"AICc = {aic}")
            print(f"Success Rate = {success_rate*100:.2f}%")
            print("=" * 20)

        return FitResult.create1param(
            A0, A0_err, m0, m0_err, redchi, aic, success_rate, method="Direct Fit"
        )
