import numpy as np
from bootstrap_fit_fixed import run_one_state_fit, run_two_state_fit

if __name__ == "__main__":
    filepath = "./data/processed/mom/bs_samples/phi_p2_0_bs.npy"

    result_one_state = run_one_state_fit(path=filepath, T=96, tmin=10, tmax=30)
    # result_two_state = run_two_state_fit()

    print("One-state fit result:")
    m0_list = result_one_state["m0_list"]
    A0_list = result_one_state["A0_list"]
    m0_mean = np.mean(m0_list)
    m0_std = np.std(m0_list, ddof=1)
    A0_mean = np.mean(A0_list)
    A0_std = np.std(A0_list, ddof=1)
    print(f"m0 = {m0_mean:.6f} ± {m0_std:.6f}")
    print(f"A0 = {A0_mean} ± {A0_std}")

    # print("Two-state fit result:")
