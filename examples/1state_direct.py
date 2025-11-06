"""
Description: example of using DirectFit to fit one-state model.
@author: GeorgeLiu
@since: 2025.11
"""

import numpy as np
from src.fitting.direct_fit import DirectFit
from src.models.model_functions import one_cosh_func
from src.models.residuals import make_uncorrelated_residual
from src.utils.io import load_data

data, t, _ = load_data(path="./data/raw/mass/", mumu=0)
residual = make_uncorrelated_residual(one_cosh_func)

# 选择拟合区间
mask = (t >= 5) & (t <= 16)
data_masked = data[:, mask]
t_masked = t[mask]

mean, err = data_masked.mean(axis=0), np.std(data_masked, axis=0, ddof=1) / np.sqrt(
    len(data_masked)
)

direct_fit = DirectFit(model_func=one_cosh_func, residual_func=residual)
result = direct_fit.fit(
    t_fit=t_masked, T=96, data_fit=mean, sigma=err, residual=residual, print_report=True
)
