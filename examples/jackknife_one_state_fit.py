from src.fitting.jackknife_fit import JackknifeFit
from src.models.model_functions import one_cosh_func
from src.models.residuals import make_uncorrelated_residual
from src.utils.io import load_data
from src.utils.stats import jackknife_resample

data,t,_ = load_data(path="./data/raw/mass/",mumu=0)
residual = make_uncorrelated_residual(one_cosh_func)

# 构造 jackknife 样本
mask = (t>=5) & (t<=16)
data_masked = data[:,mask]
t_masked = t[mask]
jk_samples, jk_mean,jk_err = jackknife_resample(data=data_masked,return_samples=True)

jk = JackknifeFit(model_func=one_cosh_func,residual_func=residual)
result = jk.fit(t_fit=t_masked,T=96,jk_samples=jk_samples,sigma=jk_err,residual=residual,print_report=True)

# print("拟合结果:")
# print(f"A0 = {result.params['A0'].value} ± {result.params['A0'].stderr}")
# print(f"m0 = {result.params['m0'].value} ± {result.params['m0'].stderr}")
# print(f"redchi = {result.redchi}")
# print(f"aic = {result.aic}")
# print(f"成功率 = {result.success_rate*100:.2f}%")