import numpy as np
from scipy.optimize import least_squares


# -----------------------------
#  Your functions
# -----------------------------
def get_aicc(chi2, n_data, n_param):
    """
    AICc = -2 ln(chi2) + 2*k + [2k(k+1)] / (n - k - 1)
    """
    aic = 2 * n_param - 2 * np.log(chi2)
    correction = (2 * n_param * (n_param + 1)) / (n_data - n_param - 1)
    aicc = aic + correction
    return aicc


def get_redchi2(result, n_data, n_param):
    dof = n_data - n_param
    redchi2 = np.sum(result.fun**2) / dof
    return redchi2


def get_chi2(result):
    chi2 = np.sum(result.fun**2)
    return chi2


# -----------------------------
#  True model (one cosh)
# -----------------------------
def one_cosh(params, t, T):
    A0, m0 = params
    return A0 * np.cosh(m0 * (t - T / 2))


def residuals(p, t, y, err, T):
    return (y - one_cosh(p, t, T)) / err


# -----------------------------
#  Build synthetic test data
# -----------------------------
def build_test_data():
    np.random.seed(42)

    T = 20
    t = np.arange(0, T)

    # true parameters
    A0_true = 1.0
    m0_true = 0.25

    y_true = one_cosh([A0_true, m0_true], t, T)

    # Add gaussian noise
    noise = 0.02 * np.random.randn(len(t))
    y_obs = y_true + noise

    err = 0.02 * np.ones_like(t)

    return t, y_obs, err, T


# -----------------------------
#  Main test
# -----------------------------
def main():
    t, y, err, T = build_test_data()
    n_data = len(t)
    n_param = 2

    # Fit with least_squares
    p0 = [0.8, 0.10]
    bounds = ([0, 0], [10, 1])

    result = least_squares(
        residuals, p0, args=(t, y, err, T), bounds=bounds, method="trf"
    )

    # Your calculations
    chi2 = get_chi2(result)
    redchi2 = get_redchi2(result, n_data, n_param)
    aicc = get_aicc(chi2, n_data, n_param)

    # Ground truth independent AICc calculation
    dof = n_data - n_param
    # chi2 = sum( (residual)^2 )
    chi2_gt = np.sum(result.fun**2)
    aic_gt = 2 * n_param - 2 * np.log(chi2_gt)
    aicc_gt = aic_gt + (2 * n_param * (n_param + 1)) / (n_data - n_param - 1)

    print("===================================")
    print("         Fit results")
    print("===================================")
    print(f"Fitted A0 = {result.x[0]:.6f}")
    print(f"Fitted m0 = {result.x[1]:.6f}")

    print("\n===================================")
    print("    Test of chi2 / redchi2 / AICc")
    print("===================================")
    print(f"chi2 (yours)       = {chi2:.6f}")
    print(f"chi2 (ground truth)= {chi2_gt:.6f}")

    print(f"\nredchi2            = {redchi2:.6f}")

    print(f"\nAICc (your fn)     = {aicc:.6f}")
    print(f"AICc (ground truth)= {aicc_gt:.6f}")

    print("\nDifference in AICc =", abs(aicc - aicc_gt))


if __name__ == "__main__":
    main()
