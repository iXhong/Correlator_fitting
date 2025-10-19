import direct_fit
import jackknife_fit
import numpy as np
import matplotlib.pyplot as plt 


def direct_fit_tmin_scan(tmin_left,tmin_right,tmax,T):
    data, t, N_cfg = direct_fit.data_load()
    mean_corr = np.mean(data, axis=0)
    mass_list = []
    mass_err_list = []
    tmin_list = np.arange(tmin_left,tmin_right+1,1)

    for tmin in tmin_list:
        result,*_ = direct_fit.direct_fit(t_min=tmin, t_max=tmax, t=t, data=data,T=T,print_report=False)
        mass_list.append(result.params['m0'].value)
        mass_err_list.append(result.params['m0'].stderr)

    return tmin_list,mass_list,mass_err_list


def jackknife_fit_tmin_scan(tmin_left,tmin_right,tmax,T):

    mass_list = []
    mass_err_list = []
    redchi_list = []
    aicc_list = []
    tmin_list = np.arange(tmin_left,tmin_right+1,1)

    for tmin in tmin_list:
        # out = jackknife_fit.main(tmin=tmin,tmax=tmax,T=96)
        out = jackknife_fit.main(tmin=tmin,tmax=tmax,show_plot=False,print_report=False)
        mass_list.append(out.params['m0'].value)
        mass_err_list.append(out.params['m0'].stderr)
        redchi_list.append(out.redchi)
        aicc_list.append(out.aicc)
    
    # np.savez("varying_tmin.txt",tmin=tmin_list,m0=m0_j,err=m0_err_j)

    return tmin_list,mass_list,mass_err_list,redchi_list,aicc_list

def compare_plot_dualy(tmax):
    data1 = np.load(f"direct_tmin_scan_{tmax}.npz")
    t_list1, m0_1, m0_err_1 = data1['tmin'], data1['m0'], data1['err']

    data2 = np.load(f"jk_tmin_scan_{tmax}.npz")
    t_list2, m0_2, m0_err_2 = data2['tmin'], data2['m0'], data2['err']
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 第一个y轴（左轴）- 绘制data1
    color1 = 'tab:blue'
    ax1.set_xlabel(r"$n_{min}$")
    ax1.set_ylabel('m0 (direct)', color=color1)
    line1 = ax1.errorbar(t_list1, m0_1, yerr=m0_err_1, fmt='s--', 
                        color=color1, capsize=3, alpha=0.7, label='direct_tmin_scan')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 创建第二个y轴（右轴）- 绘制data2
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('m0 (jk)', color=color2)
    line2 = ax2.errorbar(t_list2, m0_2, yerr=m0_err_2, fmt='o-', 
                        color=color2, capsize=3, alpha=0.7, label='jk_tmin_scan')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title(f"tmin scan, tmax={tmax} (Dual Y-axis)")
    plt.tight_layout()
    plt.show()

def compare_plot(tmax):
    data1 = np.load(f"direct_tmin_scan_{tmax}.npz")
    t_list1, m0_1, m0_err_1 = data1['tmin'], data1['m0'], data1['err']

    data2 = np.load(f"jk_tmin_scan_{tmax}.npz")
    t_list2, m0_2, m0_err_2 = data2['tmin'], data2['m0'], data2['err']

    plt.figure(figsize=(10, 6))
    plt.errorbar(t_list1, m0_1, yerr=m0_err_1, fmt='s--', color='blue', 
                capsize=3, alpha=0.5, label='direct_tmin_scan')
    plt.errorbar(t_list2, m0_2, yerr=m0_err_2, fmt='o-', color='red', 
                capsize=3, alpha=0.5, label='jk_tmin_scan')

    plt.title(f"tmin scan, tmax={tmax}")
    plt.xlabel(r"$n_{min}$")
    plt.ylabel("m0")
    plt.legend()
    
    # 更细的网格设置
    plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.4)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
    plt.minorticks_on()  # 开启次刻度
    
    plt.show()

def plot(type):
    """
    Plots data from 'jk_tmin_scan.npz' on two subplots of a single figure.
    One subplot has a free y-axis scale, the other has a fixed y-axis scale (-25, 30).
    """
    data = np.load("jk_tmin_scan.npz")
    tmin_list, y = data['tmin'], data[f'{type}']

    # 创建一个包含两个子图的图形，布局为 2 行 1 列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10)) # 可以根据需要调整 figsize

    # 第一个子图 (ax1) - 不限制 y 轴范围
    ax1.scatter(tmin_list, y, label=f"{type}")
    ax1.set_xlabel(r"$n_{min}$")  # x 轴标签通常只需要在最下面的子图或每个子图都加
    ax1.set_ylabel(f"{type}")     # 添加 y 轴标签以区分子图
    ax1.set_title("Unrestricted Y-axis") # 添加标题
    ax1.legend()
    # ax1 使用默认的 y 轴范围（由数据决定）

    # 第二个子图 (ax2) - 限制 y 轴范围为 -25 到 30
    ax2.scatter(tmin_list, y, label=f"{type}")
    ax2.set_xlabel(r"$n_{min}$")
    ax2.set_ylabel(f"{type}")     # 添加 y 轴标签
    ax2.set_title("Restricted Y-axis (-25, 30)") # 添加标题
    ax2.set_ylim(-25, 30)  # 应用 y 轴限制
    ax2.legend()

    # 调整子图之间的间距，防止重叠
    plt.tight_layout()

    # 显示整个图形
    plt.savefig("jk_1_fit_aicc.png",dpi=300)
    plt.show()


def mass_dual_y():

    data2 = np.load("jk_tmin_scan.npz")
    t_list2, m0_2, m0_err_2,redchi2,aicc2 = data2['tmin'], data2['m0'], data2['err'],data2['redchi'],data2['aicc']
    
    fig,ax1 = plt.subplots()
    lines1 = ax1.errorbar(t_list2, m0_2, yerr=m0_err_2, fmt='o-', color='red', 
                capsize=3, alpha=0.5, label=r'$am_0$')
    ax1.set_ylabel(r"$am_0$")
    ax2 = ax1.twinx()
    lines2 = ax2.scatter(t_list2,redchi2,label="redchi2")
    ax2.set_ylabel("redchi2")
    ax2.hlines(1,xmin=1,xmax=24,linestyles='dashed',label="redchi2=1")
    ax2.set_ylim(-5,5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.savefig("redchi2_m_jk_1.png",dpi=300)
    plt.show()


if __name__ == "__main__":

    mass_dual_y()

    # tmin_list,m0_d,m0_err_d = direct_fit_tmin_scan(tmin_left=1,tmin_right=24,tmax=48,T=96)
    # tmin_list ,m0_j,m0_err_j,redchi_j,aicc_j = jackknife_fit_tmin_scan(tmin_left=1,tmin_right=24,tmax=48,T=96)

    # np.savez("jk_tmin_scan",tmin=tmin_list,m0=m0_j,err=m0_err_j,redchi=redchi_j,aicc=aicc_j)
    # np.savez("direct_tmin_scan",tmin=tmin_list,m0=m0_d,err=m0_err_d)

    # plt.figure()
    # plt.errorbar(tmin_list,m0_d,m0_err_d,fmt='o',linestyle='-')
    # plt.errorbar(tmin_list,m0_j,m0_err_j,fmt='o',linestyle='-')
    # # plt.errorbar(tmax_list,m0_j,m0_err_j,fmt='o',linestyle='-')
    # plt.title(f"tmax={24}")
    # plt.ylim(0.5,0.7)
    # plt.show()

    # compare_plot(48)
    # compare_plot_dualy(48)

    # data1 = np.load("direct_tmin_scan.npz")
    # t_list1, m0_1, m0_err_1 = data1['tmin'], data1['m0'], data1['err']

    data2 = np.load("jk_tmin_scan.npz")
    t_list2, m0_2, m0_err_2,redchi2 = data2['tmin'], data2['m0'], data2['err'],data2['redchi']

    plt.figure(figsize=(10, 6))
    # plt.errorbar(t_list1, m0_1, yerr=m0_err_1, fmt='s--', color='blue', 
    #             capsize=3, alpha=0.5, label='direct_tmin_scan')
    plt.errorbar(t_list2, m0_2, yerr=m0_err_2, fmt='o-', color='red', 
                capsize=3, alpha=0.5, label='jk_tmin_scan')
    plt.scatter(t_list2,redchi2,)
    
    # plt.hlines(0.6,xmin=1,xmax=24,colors='black',linestyles='--')

    # plt.title("tmin scan, tmax=48")
    # # plt.ylim(0,2)
    # plt.xlabel(r"$n_{min}$")
    # plt.ylabel("m0")
    # plt.legend()
    
    # # 更细的网格设置
    # plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.4)
    # plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
    # plt.minorticks_on()  # 开启次刻度
    # # plt.savefig("tmin_scan_compare.png",dpi=300)
    # plt.show()