import matplotlib.pyplot as plt
import numpy as np

# 模拟总体和样本
np.random.seed(42)
population = np.random.normal(170, 8, 10000)  # 10000人的真实身高
sample = np.random.choice(population, 50, replace=False)  # 随机抽50人

true_mean = np.mean(population)
estimated_mean = np.mean(sample)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(population, bins=50, alpha=0.7, label='总体')
plt.axvline(true_mean, color='red', linewidth=2, label=f'真实均值: {true_mean:.2f}')
plt.title('总体分布 (我们看不到)')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(sample, bins=20, alpha=0.7, color='orange', label='样本')
plt.axvline(estimated_mean, color='blue', linewidth=2, label=f'估计均值: {estimated_mean:.2f}')
plt.axvline(true_mean, color='red', linestyle='--', label=f'真实均值: {true_mean:.2f}')
plt.title('样本分布 (我们能看到)')
plt.legend()

plt.tight_layout()
plt.show()

print(f"真实均值: {true_mean:.2f}")
print(f"估计均值: {estimated_mean:.2f}")
print(f"估计误差: {abs(estimated_mean - true_mean):.2f}")