"""
scan t_max with a fixed t_min
@author: George Liu
@since 2025.11
"""
import numpy as np
import matplotlib.pyplot as plt
from src.utils import io
from src.models.residuals import make_uncorrelated_residual
from src.fitting.two_state_fit import one_state_fit

def jackknife_fit_tmax_scan(tmin,tmax_left,tmax_right,T):

