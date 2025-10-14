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
```

### 3.内容

#### 3.1 简单的direct fit + uncorrelated 拟合 大致思路
1. First thing first, 先观察一下你拿到的关联函数数据，画一下correlator 的图像。
2. 对数据可以做**对折平均**（对一半的数据拟合），**缩放**（对放大之后的数据拟合）
3. 计算有效质量effective mass,
4. 从预处理后的correlator中计算y_mean和y_error
5. 利用**最小二乘法**做拟合，我们要minimize的函数是residual
   $$
   \text{residual} = \frac{y_{model} - y_{data}}{y_{error}}
   $$
6. 推荐[LMfit](https://github.com/lmfit/lmfit-py)来做最小化拟合，拟合算法使用`least_squares`或者`leastsq`
7. 拟合的初始值设定时，`m0`可以使用*有效质量*
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
- [ ] 变化拟合范围，尝试选择最好的拟合（select the best fit）


### 5.Bugs

- [] jackknife fit 的实现存在严重的问题 (似乎已经解决)




### 6.Reference
1. Jackknife and Bootstrap Resampling Methods in Statistical Analysis to Correct for Bias
2. Hadronic correlators from heavy to very  light quarks  Spectral and transport properties from lattice QCD
3. 淬火近似下重夸克偶素热修正的格点量子色动力学研究
