import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')

class CorrelatorAnalysis:
    def __init__(self, data):
        """
        初始化关联函数分析类
        data: shape (Ncfg, T) 的二维数组
        """
        self.data = data
        self.Ncfg, self.T = data.shape
        self.t_array = np.arange(self.T)
        
        # 计算基本统计量
        self.mean_corr = np.mean(data, axis=0)
        self.std_corr = np.std(data, axis=0, ddof=1)
        self.err_corr = self.std_corr / np.sqrt(self.Ncfg)
        
    def jackknife_resample(self):
        """步骤2: 用jackknife建立协方差矩阵"""
        jackknife_samples = []
        for i in range(self.Ncfg):
            # 删除第i个样本
            mask = np.ones(self.Ncfg, dtype=bool)
            mask[i] = False
            jk_sample = np.mean(self.data[mask], axis=0)
            jackknife_samples.append(jk_sample)
        
        self.jackknife_samples = np.array(jackknife_samples)
        
        # 计算协方差矩阵
        self.cov_matrix = np.zeros((self.T, self.T))
        for i in range(self.T):
            for j in range(self.T):
                diff_i = self.jackknife_samples[:, i] - self.mean_corr[i]
                diff_j = self.jackknife_samples[:, j] - self.mean_corr[j]
                self.cov_matrix[i, j] = (self.Ncfg - 1) / self.Ncfg * np.sum(diff_i * diff_j)
        
        return self.cov_matrix
    
    def effective_mass(self):
        """步骤3: 计算有效质量"""
        self.m_eff = np.zeros(self.T - 1)
        
        for t in range(self.T - 1):
            if self.mean_corr[t] > 0 and self.mean_corr[t+1] > 0:
                ratio = self.mean_corr[t] / self.mean_corr[t+1]
                if ratio > 1:
                    self.m_eff[t] = np.log(ratio)
                else:
                    self.m_eff[t] = np.nan
            else:
                self.m_eff[t] = np.nan
        
        return self.m_eff
    
    def fit_function(self, params, t):
        """拟合函数: A0 * cosh(-m0 * (t - T/2))"""
        A0 = params['A0']
        m0 = params['m0']
        return A0 * np.cosh(-m0 * (t - self.T/2))
    
    def residual(self, params, t_fit, data_fit, inv_cov):
        """残差函数，用于lmfit minimize"""
        model = self.fit_function(params, t_fit)
        residual = data_fit - model
        # 使用协方差矩阵加权
        chi2 = residual.T @ inv_cov @ residual
        return np.sqrt(chi2)
    
    def single_state_fit(self, t_min=5, t_max=None):
        """步骤4: 单态拟合"""
        if t_max is None:
            t_max = self.T - t_min
        
        # 选择拟合区间
        fit_mask = (self.t_array >= t_min) & (self.t_array <= t_max)
        t_fit = self.t_array[fit_mask]
        data_fit = self.mean_corr[fit_mask]
        
        # 构建协方差矩阵的子矩阵
        cov_fit = self.cov_matrix[np.ix_(fit_mask, fit_mask)]
        
        # 添加正则化避免奇异矩阵
        reg_param = 1e-10 * np.trace(cov_fit) / len(cov_fit)
        cov_fit += reg_param * np.eye(len(cov_fit))
        
        try:
            inv_cov = np.linalg.inv(cov_fit)
        except np.linalg.LinAlgError:
            # 如果协方差矩阵奇异，使用对角近似
            inv_cov = np.diag(1.0 / np.diag(cov_fit))
        
        # 设置初始参数
        params = Parameters()
        params.add('A0', value=data_fit[0], min=0)
        params.add('m0', value=0.5, min=0.01, max=2.0)
        
        # 执行拟合
        result = minimize(self.residual, params, args=(t_fit, data_fit, inv_cov))
        
        # 计算卡方和自由度
        model_fit = self.fit_function(result.params, t_fit)
        residual_vec = data_fit - model_fit
        chi2 = residual_vec.T @ inv_cov @ residual_vec
        dof = len(t_fit) - len(result.params)
        
        self.single_fit_result = {
            'params': result.params,
            'chi2': chi2,
            'dof': dof,
            'chi2_per_dof': chi2/dof if dof > 0 else np.inf,
            't_range': (t_min, t_max),
            'success': result.success
        }
        
        return self.single_fit_result
    
    def two_state_fit(self, t_min=3, t_max=None):
        """步骤5: 双态拟合"""
        if not hasattr(self, 'single_fit_result'):
            raise ValueError("请先执行单态拟合")
        
        if t_max is None:
            t_max = self.T - t_min
            
        # 双态拟合函数
        def two_state_function(params, t):
            A0 = params['A0']
            m0 = params['m0']
            A1 = params['A1']
            m1 = params['m1']
            return (A0 * np.cosh(-m0 * (t - self.T/2)) + 
                   A1 * np.cosh(-m1 * (t - self.T/2)))
        
        def two_state_residual(params, t_fit, data_fit, inv_cov):
            model = two_state_function(params, t_fit)
            residual = data_fit - model
            chi2 = residual.T @ inv_cov @ residual
            return np.sqrt(chi2)
        
        # 选择拟合区间
        fit_mask = (self.t_array >= t_min) & (self.t_array <= t_max)
        t_fit = self.t_array[fit_mask]
        data_fit = self.mean_corr[fit_mask]
        
        cov_fit = self.cov_matrix[np.ix_(fit_mask, fit_mask)]
        reg_param = 1e-10 * np.trace(cov_fit) / len(cov_fit)
        cov_fit += reg_param * np.eye(len(cov_fit))
        
        try:
            inv_cov = np.linalg.inv(cov_fit)
        except np.linalg.LinAlgError:
            inv_cov = np.diag(1.0 / np.diag(cov_fit))
        
        # 使用单态拟合结果作为初值
        single_params = self.single_fit_result['params']
        
        params = Parameters()
        params.add('A0', value=single_params['A0'].value, min=0)
        params.add('m0', value=single_params['m0'].value, min=0.01, max=2.0)
        params.add('A1', value=single_params['A0'].value * 0.1, min=0)
        params.add('m1', value=single_params['m0'].value * 2, min=single_params['m0'].value)
        
        result = minimize(two_state_residual, params, args=(t_fit, data_fit, inv_cov))
        
        # 计算卡方
        model_fit = two_state_function(result.params, t_fit)
        residual_vec = data_fit - model_fit
        chi2 = residual_vec.T @ inv_cov @ residual_vec
        dof = len(t_fit) - len(result.params)
        
        self.two_state_fit_result = {
            'params': result.params,
            'chi2': chi2,
            'dof': dof,
            'chi2_per_dof': chi2/dof if dof > 0 else np.inf,
            't_range': (t_min, t_max),
            'success': result.success
        }
        
        return self.two_state_fit_result
    
    def bootstrap_error_analysis(self, n_bootstrap=1000):
        """步骤6: Bootstrap误差分析"""
        if not hasattr(self, 'single_fit_result'):
            raise ValueError("请先执行拟合")
        
        t_min, t_max = self.single_fit_result['t_range']
        
        bootstrap_masses = []
        
        for _ in range(n_bootstrap):
            # Bootstrap重采样
            indices = np.random.choice(self.Ncfg, self.Ncfg, replace=True)
            boot_data = self.data[indices]
            boot_mean = np.mean(boot_data, axis=0)
            
            # 拟合区间
            fit_mask = (self.t_array >= t_min) & (self.t_array <= t_max)
            t_fit = self.t_array[fit_mask]
            data_fit = boot_mean[fit_mask]
            
            # 简化的对角协方差矩阵用于bootstrap
            boot_std = np.std(boot_data, axis=0, ddof=1)
            inv_cov_diag = np.diag(1.0 / (boot_std[fit_mask]**2 + 1e-10))
            
            # 拟合参数
            params = Parameters()
            params.add('A0', value=data_fit[0], min=0)
            params.add('m0', value=0.5, min=0.01, max=2.0)
            
            try:
                result = minimize(self.residual, params, args=(t_fit, data_fit, inv_cov_diag))
                if result.success:
                    bootstrap_masses.append(result.params['m0'].value)
            except:
                continue
        
        if len(bootstrap_masses) > 0:
            self.bootstrap_mass_error = np.std(bootstrap_masses)
            self.bootstrap_masses = np.array(bootstrap_masses)
        else:
            self.bootstrap_mass_error = np.nan
            self.bootstrap_masses = np.array([])
        
        return self.bootstrap_mass_error
    
    def plot_results(self):
        """步骤7: 绘制结果图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 关联函数
        ax1.errorbar(self.t_array, self.mean_corr, yerr=self.err_corr, 
                    fmt='o', label='Data', capsize=3)
        ax1.set_yscale('log')
        ax1.set_xlabel('t')
        ax1.set_ylabel('G(t)')
        ax1.set_title('Correlator')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 有效质量
        if hasattr(self, 'm_eff'):
            valid_mask = ~np.isnan(self.m_eff)
            ax2.plot(self.t_array[1:][valid_mask], self.m_eff[valid_mask], 'o-', label='m_eff')
            
            # 如果有拟合结果，画出拟合的质量
            if hasattr(self, 'single_fit_result'):
                m_fit = self.single_fit_result['params']['m0'].value
                ax2.axhline(y=m_fit, color='r', linestyle='--', 
                           label=f'Fitted m = {m_fit:.4f}')
        
        ax2.set_xlabel('t')
        ax2.set_ylabel('m_eff(t)')
        ax2.set_title('Effective Mass')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 拟合比较
        if hasattr(self, 'single_fit_result'):
            t_min, t_max = self.single_fit_result['t_range']
            fit_mask = (self.t_array >= t_min) & (self.t_array <= t_max)
            t_fit = self.t_array[fit_mask]
            
            ax3.errorbar(self.t_array, self.mean_corr, yerr=self.err_corr, 
                        fmt='o', label='Data', capsize=3)
            
            # 单态拟合
            model_single = self.fit_function(self.single_fit_result['params'], self.t_array)
            ax3.plot(self.t_array, model_single, 'r-', label='Single-state fit')
            
            # 双态拟合（如果存在）
            if hasattr(self, 'two_state_fit_result'):
                def two_state_function(params, t):
                    A0 = params['A0']
                    m0 = params['m0']
                    A1 = params['A1']
                    m1 = params['m1']
                    return (A0 * np.cosh(-m0 * (t - self.T/2)) + 
                           A1 * np.cosh(-m1 * (t - self.T/2)))
                
                model_two = two_state_function(self.two_state_fit_result['params'], self.t_array)
                ax3.plot(self.t_array, model_two, 'g-', label='Two-state fit')
            
            ax3.set_yscale('log')
            ax3.axvspan(t_min, t_max, alpha=0.2, color='yellow', label='Fit range')
            
        ax3.set_xlabel('t')
        ax3.set_ylabel('G(t)')
        ax3.set_title('Fit Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Bootstrap分布（如果存在）
        if hasattr(self, 'bootstrap_masses') and len(self.bootstrap_masses) > 0:
            ax4.hist(self.bootstrap_masses, bins=50, alpha=0.7, density=True)
            ax4.axvline(np.mean(self.bootstrap_masses), color='r', linestyle='--', 
                       label=f'Mean = {np.mean(self.bootstrap_masses):.4f}')
            ax4.axvline(np.mean(self.bootstrap_masses) + self.bootstrap_mass_error, 
                       color='r', linestyle=':', alpha=0.7)
            ax4.axvline(np.mean(self.bootstrap_masses) - self.bootstrap_mass_error, 
                       color='r', linestyle=':', alpha=0.7)
            ax4.set_xlabel('Mass')
            ax4.set_ylabel('Density')
            ax4.set_title('Bootstrap Mass Distribution')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No bootstrap data', ha='center', va='center', 
                    transform=ax4.transAxes)
            ax4.set_title('Bootstrap Analysis')
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self):
        """打印分析结果"""
        print("="*60)
        print("关联函数分析结果")
        print("="*60)
        
        print(f"数据统计:")
        print(f"  配置数: {self.Ncfg}")
        print(f"  时间长度: {self.T}")
        
        if hasattr(self, 'single_fit_result'):
            result = self.single_fit_result
            print(f"\n单态拟合结果:")
            print(f"  拟合区间: t = {result['t_range'][0]} - {result['t_range'][1]}")
            print(f"  A0 = {result['params']['A0'].value:.6f} ± {result['params']['A0'].stderr or 0:.6f}")
            print(f"  m0 = {result['params']['m0'].value:.6f} ± {result['params']['m0'].stderr or 0:.6f}")
            print(f"  χ²/dof = {result['chi2_per_dof']:.3f}")
            print(f"  拟合成功: {result['success']}")
        
        if hasattr(self, 'two_state_fit_result'):
            result = self.two_state_fit_result
            print(f"\n双态拟合结果:")
            print(f"  拟合区间: t = {result['t_range'][0]} - {result['t_range'][1]}")
            print(f"  A0 = {result['params']['A0'].value:.6f} ± {result['params']['A0'].stderr or 0:.6f}")
            print(f"  m0 = {result['params']['m0'].value:.6f} ± {result['params']['m0'].stderr or 0:.6f}")
            print(f"  A1 = {result['params']['A1'].value:.6f} ± {result['params']['A1'].stderr or 0:.6f}")
            print(f"  m1 = {result['params']['m1'].value:.6f} ± {result['params']['m1'].stderr or 0:.6f}")
            print(f"  χ²/dof = {result['chi2_per_dof']:.3f}")
        
        if hasattr(self, 'bootstrap_mass_error'):
            print(f"\nBootstrap误差分析:")
            print(f"  质量 (Bootstrap): {np.mean(self.bootstrap_masses):.6f} ± {self.bootstrap_mass_error:.6f}")

# 使用示例
def main():
    # 假设您的数据已经加载为 data，形状为 (Ncfg, T)
    # data = np.load('your_correlator_data.npy')  # 替换为您的数据文件
    
    # 这里创建一些示例数据用于演示
    Ncfg, T = 1000, 32
    t_array = np.arange(T)
    
    # 模拟关联函数数据 A*cosh(-m*(t-T/2)) + noise
    A_true, m_true = 1.0, 0.5
    true_corr = A_true * np.cosh(-m_true * (t_array - T/2))
    
    # 添加噪声
    np.random.seed(42)
    noise_level = 0.1
    data = true_corr[None, :] + noise_level * np.random.randn(Ncfg, T) * true_corr[None, :]
    
    print("开始分析关联函数数据...")
    
    # 创建分析对象
    analyzer = CorrelatorAnalysis(data)
    
    # 步骤1-2: 计算统计量和协方差矩阵
    print("步骤1-2: 计算统计量和Jackknife协方差矩阵...")
    analyzer.jackknife_resample()
    
    # 步骤3: 计算有效质量
    print("步骤3: 计算有效质量...")
    analyzer.effective_mass()
    
    # 步骤4: 单态拟合
    print("步骤4: 执行单态拟合...")
    analyzer.single_state_fit(t_min=8, t_max=24)
    
    # 步骤5: 双态拟合
    print("步骤5: 执行双态拟合...")
    try:
        analyzer.two_state_fit(t_min=5, t_max=27)
    except Exception as e:
        print(f"双态拟合失败: {e}")
    
    # 步骤6: Bootstrap误差分析
    print("步骤6: Bootstrap误差分析...")
    analyzer.bootstrap_error_analysis(n_bootstrap=500)
    
    # 步骤7: 打印结果和绘图
    analyzer.print_results()
    analyzer.plot_results()

if __name__ == "__main__":
    main()


