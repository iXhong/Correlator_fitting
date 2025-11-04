"""
residual functions
@George Liu
@since 2025
"""

import numpy as np
from functools import wraps


def make_uncorrelated_residual(model_func):
    @wraps(model_func)
    def uncorrelated_residual(params, t_fit, data_fit, err_fit, T):
        """带权重的不相关残差函数"""
        return (data_fit - model_func(params, t_fit, T)) / err_fit  # 用误差加权

    return uncorrelated_residual


def correlated_residual(params, t_fit, data_fit, err_fit, model_func, T):
    """带权重的相关残差函数
    考虑完整的协方差矩阵
    """


if __name__ == "__main__":
    pass
