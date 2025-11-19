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
根据反周期性边界条件做fold & average
↓
得到 per-config 的 G(p²,t)
↓
保存到文件
↓
对所有的.dat做上面的处理，
得到新的关于(p^2,t)的关联函数数据文件
(上述功能在load_data.py中实现）

───────────── all dat （所有配置） ─────────────

stack 所有 config 的 G(p²,t)
↓
bootstrap / jackknife
↓
得到samples, mean, error
↓
绘制errorbar of G
↓
effective mass
↓
...
```
