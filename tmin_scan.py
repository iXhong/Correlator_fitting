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
    tmin_list = np.arange(tmin_left,tmin_right+1,1)

    for tmin in tmin_list:
        # out = jackknife_fit.main(tmin=tmin,tmax=tmax,T=96)
        out = jackknife_fit.main(tmin=tmin,tmax=tmax,show_plot=False,print_report=False)
        mass_list.append(out.params['m0'].value)
        mass_err_list.append(out.params['m0'].stderr)
        redchi_list.append(out.redchi)
    
    # np.savez("varying_tmin.txt",tmin=tmin_list,m0=m0_j,err=m0_err_j)

    return tmin_list,mass_list,mass_err_list,redchi_list

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

def redchi_plot():
    data = np.load("jk_tmin_scan.npz")
    tmin_list,redchi = data['tmin'], data['redchi']

    plt.figure()
    plt.scatter(tmin_list,redchi)
    plt.show()



if __name__ == "__main__":
    
    # plot_d()
    # plotj()
    redchi_plot()

    # tmin_list,m0_d,m0_err_d = direct_fit_tmin_scan(tmin_left=1,tmin_right=24,tmax=48,T=96)
    # tmin_list ,m0_j,m0_err_j,redchi_j = jackknife_fit_tmin_scan(tmin_left=1,tmin_right=24,tmax=48,T=96)

    # np.savez("jk_tmin_scan",tmin=tmin_list,m0=m0_j,err=m0_err_j,redchi=redchi_j)
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

    # data2 = np.load("jk_tmin_scan.npz")
    # t_list2, m0_2, m0_err_2 = data2['tmin'], data2['m0'], data2['err']

    # plt.figure(figsize=(10, 6))
    # plt.errorbar(t_list1, m0_1, yerr=m0_err_1, fmt='s--', color='blue', 
    #             capsize=3, alpha=0.5, label='direct_tmin_scan')
    # plt.errorbar(t_list2, m0_2, yerr=m0_err_2, fmt='o-', color='red', 
    #             capsize=3, alpha=0.5, label='jk_tmin_scan')
    
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