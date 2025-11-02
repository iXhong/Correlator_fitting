"""
Description: Base classes and data structures for fitting methods.
@author: George Liu
@since: 2025.11
"""

# src/fits/base_fit.py
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable
from lmfit import Parameters


@dataclass
class Parameter:
    value: float
    stderr: float

@dataclass
class FitResult:
    params: Dict[str, Parameter] = None
    redchi: Optional[float] = None
    aic: Optional[float] = None
    success_rate: Optional[float] = None
    method: Optional[str] = ""

    @classmethod
    def create1param(cls,A0,A0_err,m0,m0_err,redchi,aic,success_rate,method):
        #create FitResult for one state fit
        params = {
            "A0": Parameter(value=A0, stderr=A0_err),
            "m0": Parameter(value=m0, stderr=m0_err)
        }
        return cls(params=params, redchi=redchi, aic=aic, success_rate=success_rate, method=method)

    @classmethod
    def create2param(cls,A0,A0_err,A1,A1_err,m0,m0_err,m1,m1_err,redchi,aic,success_rate,method):
        #create FitResult for two state fit
        params = {
            "A0": Parameter(value=A0, stderr=A0_err),
            "A1": Parameter(value=A1, stderr=A1_err),
            "m0": Parameter(value=m0, stderr=m0_err),
            "m1": Parameter(value=m1,stderr=m1_err)
        }
        return cls(params=params, redchi=redchi, aic=aic, success_rate=success_rate, method=method)


class FitMethod(ABC):
    # fitting method base class
    def __init__(self, model_func: Callable, residual_func: Callable):
        self.model_func = model_func
        self.residual_func = residual_func

    def fit(self, *args, **kwargs) -> FitResult:
        start = time.time()
        result = self._run_fit(*args, **kwargs)
        result.runtime = time.time() - start
        return result

    @abstractmethod
    def _run_fit(self, *args, **kwargs) -> FitResult:
        # main fit routine
        pass

    @abstractmethod
    def _init_params(self, data_fit)->Parameters:
        # initialize fit parameters
        pass