"""
Description:scan a range of tmax with a fixed tmin
Author:@George Liu
Since:2025.10
"""

import direct_fit
import jackknife_fit
import bootstrap_fit
import numpy as np
import matplotlib.pyplot as plt 


def jackknife_fit_tmax_scan(tmin,tmax_left,tmax_right,T):

    mass_list = []
    mass_err_list = []
    tmax_list = np.arange(tmax_left,tmax_right+1,1)

    for tmax in tmax_list:
        # out = jackknife_fit.main(tmin=tmin,tmax=tmax,T=96)
        out = jackknife_fit.main(tmin=tmin,tmax=tmax,show_plot=False,print_report=False)
        mass_list.append(out.params['m0'].value)
        mass_err_list.append(out.params['m0'].stderr)

    # np.savez("varying_tmax",tmax=tmax_list,m0=mass_list,err=mass_err_list)

    # return tmax_list,mass_list,mass_err_list
    return {
        "t_list":tmax_list,
        "m0": mass_list,
        "err":mass_err_list
    }


def direct_fit_tmax_scan(tmin,tmax_left,tmax_right,T):
    data, t, N_cfg = direct_fit.data_load()
    mass_list = []
    mass_err_list = []
    tmax_list = np.arange(tmax_left,tmax_right+1,1)

    for tmax in tmax_list:
        result,*_ = direct_fit.direct_fit(t_min=tmin,t_max=tmax,t=t,data=data,T=T,print_report=False)
        mass_list.append(result.params['m0'].value)
        mass_err_list.append(result.params['m0'].stderr)

    # return tmin_list,mass_list,mass_err_list
    return {
        "t_list":tmax_list,
        "m0":mass_list,
        "err":mass_err_list
    }


def bootstrap_fit_tmax_scan(tmin,tmax_left,tmax_right,n_resamples):

    mass_list = []
    mass_err_list = []
    tmax_list = np.arange(tmax_left,tmax_right+1,1)

    for tmax in tmax_list:
        # out = jackknife_fit.main(tmin=tmin,tmax=tmax,T=96)
        out = bootstrap_fit.main(tmin=tmin,tmax=tmax,show_plot=False,print_report=False,n_resamples=n_resamples)
        mass_list.append(out.params['m0'].value)
        mass_err_list.append(out.params['m0'].stderr)

    # np.savez("varying_tmax",tmax=tmax_list,m0=mass_list,err=mass_err_list)

    # return tmax_list,mass_list,mass_err_list
    return {
        "t_list":tmax_list,
        "m0": mass_list,
        "err":mass_err_list
    }


if __name__ == "__main__":

    # result = direct_fit_tmax_scan(tmin=5,tmax_left=10,tmax_right=48,T=96)
    resultj = jackknife_fit_tmax_scan(tmin=5,tmax_left=10,tmax_right=48,T=96)
    resultb = bootstrap_fit_tmax_scan(tmin=5,tmax_left=10,tmax_right=48,n_resamples=20)

    t_list = resultb["t_list"]
    m0_b = resultb["m0"]
    m0_errb = resultb['err']
    m0_j = resultj['m0']
    m0_errj = resultj['err']

    # np.savez("jk_tmax_scan",t_list=t_list,m0=m0,m0_err=m0_err)

    # data = np.load("jk_tmax_scan.npz")
    # t_list,m0,m0_err = data['t_list'],data['m0'],data['m0_err']
    plt.figure()
    plt.errorbar(t_list, y=m0_j, yerr=m0_errj, fmt='s--', capsize=3,alpha=0.7,label='jackknife')
    plt.errorbar(t_list,y=m0_b,yerr=m0_errb,fmt='s--',capsize=3,alpha=0.5,label='bootstrap')
    plt.title("tmax scan,tmin=5")
    plt.xlabel(r"$n_{max}$")
    # plt.ylim(0.5, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


    #读取npz文件作图
    # data1 = np.load("direct_tmax_scan.npz")
    # t_list1, m0_1, m0_err_1 = data1['t_list'], data1['m0'], data1['m0_err']
    
    # data2 = np.load("jk_tmax_scan.npz")
    # t_list2, m0_2, m0_err_2 = data2['t_list'], data2['m0'], data2['m0_err']

    # plt.figure()
    # plt.errorbar(t_list1, y=m0_1, yerr=m0_err_1, fmt='s--', capsize=3,alpha=0.5,c='r',label='direct fit 1state')
    # plt.errorbar(t_list2, y=m0_2, yerr=m0_err_2, fmt='s--', capsize=3,alpha=0.5,c='b',label='jk fit 1state')
    # plt.title("tmax scan,tmin=5")
    # plt.xlabel(r"$n_{max}$")
    # plt.legend()
    # # plt.ylim(0.5, 1)
    # plt.grid(True, linestyle='--', alpha=0.5)
    # # plt.savefig("tmax_scan_compare.png",dpi=300)
    # plt.show()


    # fig, ax1 = plt.subplots(figsize=(10, 6))

    # # 第一个y轴（左轴）- 绘制data1
    # color1 = 'tab:blue'
    # ax1.set_xlabel(r"$n_{max}$")
    # ax1.set_ylabel('m0 (direct)', color=color1)
    # line1 = ax1.errorbar(t_list1, m0_1, yerr=m0_err_1, fmt='s--', 
    #                     color=color1, capsize=3, alpha=0.7, label='direct_tmax_scan')
    # ax1.tick_params(axis='y', labelcolor=color1)
    # ax1.grid(True, linestyle='--', alpha=0.5)

    # # 创建第二个y轴（右轴）- 绘制data2
    # ax2 = ax1.twinx()
    # color2 = 'tab:red'
    # ax2.set_ylabel('m0 (jk)', color=color2)
    # line2 = ax2.errorbar(t_list2, m0_2, yerr=m0_err_2, fmt='o-', 
    #                     color=color2, capsize=3, alpha=0.7, label='jk_tmax_scan')
    # ax2.tick_params(axis='y', labelcolor=color2)

    # # 添加图例
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # plt.title("tmax scan, tmin=5 (Dual Y-axis)")
    # plt.tight_layout()
    # plt.show()




