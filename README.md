## Correlator fitting （如何拟合两点关联函数）

### 1.Overview
这个仓库主要是我学习如何拟合两点关联函数的记录


### 2.Content
各个文件分别是：
```
mass/                       格点生成的关联函数数据，15个Configration,96个时间点
direct_fit.py               直接对cosh型函数做单态拟合
jackknife_fit.py            做jackknife fit
two_state_fit.py            使用cosh型函数，做双态拟合
fit_result_plot.py          比较单态和双态拟合的结果的绘图程序
scan_tmin.py                固定tmax,扫描一系列tmin做拟合
scan_tmax.py                固定tmin,扫描一系列tmax做拟合
jk_2_cosh_fit.py            jackknife 2 state fit
bs_2_cosh_fit.py            bootstrap 2 state fit
```

### 3.内容

#### 3.1 简单的direct fit + uncorrelated 拟合 大致思路
1. First thing first, 先观察一下你拿到的关联函数数据，画一下correlator 的图像。
2. 对数据可以做**对折平均**（对一半的数据拟合），**缩放**（对放大之后的数据拟合）
3. 计算有效质量effective mass,
4. 从预处理后的correlator中计算y_mean和y_error
5. 利用**最小二乘法**做拟合，我们要minimize的函数是residual
   $$\text{residual} = \frac{y_{model} - y_{data}}{y_{error}}`$$
6. 推荐[LMfit](https://github.com/lmfit/lmfit-py)来做最小化拟合，拟合算法使用`least_squares`或者`leastsq`
7. 拟合的初始值设定时，`m0`可以使用*effective mass*
8. 不出问题，应该能成功✌。

### 3.2改变拟合区间
```bash
varying_tmax_5                tmax从10-24,tmin=5
varying_tmin_48               tmin从1-20，tmax=48
```
从拟合图像上来看,改变tmax似乎对与拟合结果改变不大,而改变tmin则会明显的影响拟合结果,而且jackknife fit和direct fit给出的结果似乎十分接近,除了前者给出的误差更大以外.

<img src="imgs/tmax_scan_compare.png" alt="tmax scan" width="500" height="400">
<img src="imgs/tmin_scan_compare.png" alt="tmax scan" width="500" height=auto>

### 4.Todo

- [x] 实现jackknife与bootstrap resample method
- [x] 对exp形式函数单态拟合
- [x] 对exp形式函数双态拟合
- [x] 对cosh形式函数单态拟合
- [x] 对cosh形式函数双态拟合
- [ ] 变化拟合范围，绘制不同tmin下的拟合结果am0,chi2/dof以及AICC,尝试选择最好的拟合（select the best fit）
- [ ] 使用bootstrap做拟合,并与jackknife 拟合做比较
- [ ] jackknife/bootstrap fit 进阶
  - [ ] correlated fit
  - [ ] two state fit
- [ ] 推导一下direct fit 和 jackknife fit 的误差的关系


### 5.Bugs

- [x] jackknife fit 的实现存在严重的问题  fixed
- [x] data_load() 中对原始的correlator数据进行对折取平均的时候出错[7.1](#71-对correlator-取平均)
- [ ] 在Analysistoolbox中jackknife与bootstrap的使用上出错


### 6.Reference
1. Jackknife and Bootstrap Resampling Methods in Statistical Analysis to Correct for Bias
2. Hadronic correlators from heavy to very  light quarks  Spectral and transport properties from lattice QCD
3. 淬火近似下重夸克偶素热修正的格点量子色动力学研究

--- 
### 7. 备注

#### 7.1 对correlator 取平均
目前我的correlator是T=96,序列是(0-95),我原来的错误处理方法是直接对折取平均,导致0和95,1和94,...,47和48取平均,这么做是错的.
在格点 QCD 或类似时间相关关联函数分析中，常见的时间关联函数
$$C(t)=⟨O(t)O†(0)⟩$$
满足周期性边界条件（periodic BC）：
$$C(t)=C(T−t)$$
这里T是时间方向的总长度（例如T=96）。

因此我们在分析时，可以只保留一半（0 到 T/2）区间的数据，因为另一半是镜像。
对称化（folding）就是取
$$C_{fold}(t)=\frac{1}{2}[C(t)+C(T−t)]$$
所以$C_{fold}(0)=\frac{1}{2}[C(0)+C(96−0)]$,所以0应该和96对应,可以想象为一个圆环,首尾相接,所以96号点就是0号点,C(t=0)与他自己对应而不是C(t=95),最终剩下t=48时,T-t = 48,所以t=48也是和自己对应.

#### 7.2 bootstrap_fit 下的估计
在实际测试中我发现使用bootstrap_fit 会出现$\text{tmax}$取某些区间的值时,对m0的估计出现很大的偏差,(这里的偏差是相较于eff_mass和jackknife_fit结果而言的).原先的代码中我对m0的估计是对bootstrap resamples得到的样本做拟合得到的结果$a_i$取平均值,但是通过绘制$a_i$的直方分布图,我发现在$m_0=0$处也出现了一个频率上的高峰,另一个高峰是在$m_0=0.65$附近,所以在这里我们似乎不能再使用取平均的方法来估计我们想要的m0的值,而应该使用**中位数**作为估计值.