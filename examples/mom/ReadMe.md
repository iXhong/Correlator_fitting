# 带有动量的两点关联函数数据处理

### 1.数据预处理 pipeline

```bash
───────────── 每个配置（单独文件） ─────────────

TwoPt.dat ───→ 解析(px,py,pz,mumu)
↓
p² binning（合并所有产生相同 p² 的动量）
↓
mumu 平均（1,2,3 → φ meson spatial 平均）
↓
根据反周期性边界条件做 fold & average
↓
得到 per-config 的 G(p²,t)
↓
保存到文件
↓
对所有的.dat 做上面的处理，
得到新的关于(p^2,t)的关联函数数据文件
(上述功能在 load_data.py 中实现 保存到p2_sorted/）

───────────── all dat （所有配置） ─────────────

stack 所有 config 的 G(p²,t)
↓
bootstrap / jackknife   (bs_resample.py 保存samples 到bs_samples)
↓
得到 samples, mean, error  (fold_average.py)
↓
绘制 errorbar of G
↓
effective mass of per bs samples   (effective_mass.py)
↓
plot errorbar of effective mass
↓
find the plateau & get the approximate mass
───────────── 拟合阶段 ─────────────
(bootstrap_fit_fixed.py)
计算总的(N_b,T)的 bs samples 的 Covriance Matrix/error 作为后续拟合的 sigma
↓
对每一个 bs sample 做单态拟合,利用每一次拟合得到的参数计算 y_fit,并绘制图像
↓
对每一个 bs sample 做双态拟合,利用每一次拟合得到的参数计算 y_fit,并绘制图像
↓
与 G 的 errorbar 图像对比
↓
固定拟合区间中的 tmax,扫描不同的 tmin,得到redchi2-n_{sigma,min},am_0 - n_{sigma,min}的图像
↓
计算AICc
↓
通过比较AICc选择the best fit:
通过比较不同tmin下不同的拟合模型的拟合的AICc的值,选择出这一个tmin下,最好的那个拟合模型
↓
画出aicc_selected_am0-n_{sigma,min}
↓
做plateau average:
通过am_0 - n_{sigma,min}的图像选择出am0在tmin变动时的平台区间,
我们认为这个平台区间内的tmin都是合理的,都是可以采用的,然后我们利用加权平均,
计算出最终的mean,sys_err,stat_err
```
