import numpy as np
import matplotlib.pyplot as plt
from bootstrap_fit_fixed import run_one_state_fit, run_two_state_fit


def main():
    # 用户配置
    BS_FILE = "./data/processed/mom/bs_samples/phi_p2_0_bs.npy"
    LATTICE_T = 96
    TMIN = 2
    TMAX = 30

    # # 运行 one-state 拟合
    # fit_results = run_one_state_fit(
    #     path=BS_FILE,
    #     T=LATTICE_T,
    #     tmin=TMIN,
    #     tmax=TMAX,
    #     savepath=None,
    # )

    # 运行 two-state 拟合
    fit_results = run_two_state_fit(
        path=BS_FILE,
        T=LATTICE_T,
        tmin=TMIN,
        tmax=TMAX,
        savepath=None,
    )

    m0_mean = np.mean(fit_results["m0_list"])
    m0_std = np.std(fit_results["m0_list"])
    print(f"One-state fit result for tmin={TMIN}, tmax={TMAX}:")
    print(f"m0 = {m0_mean:.6f} ± {m0_std:.6f}")

    plt.figure(figsize=(8, 5))
    plt.hist(fit_results["m0_list"], bins="auto", alpha=0.7, color="blue")
    plt.title(f"Histogram of am0 (tmin={TMIN}, tmax={TMAX})")
    plt.xlabel("am0")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
