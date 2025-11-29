import numpy as np
from bootstrap_fit_fixed import prepare_p2_bs_samples


def test_prepare_p2_bs_samples(path):
    # Sample data
    # data = np.load(path)
    bs_samples, cov_matrix = prepare_p2_bs_samples(path, isCorrelated=True)
    sigma = np.std(bs_samples, axis=0, ddof=1)
    # Check if the computed sigma matches the standard deviation of the samples
    assert np.allclose(
        np.sqrt(np.diag(cov_matrix)), sigma
    ), "Sigma does not match standard deviation of samples"


if __name__ == "__main__":
    path = "./data/processed/mom/bs_samples/phi_p2_0_bs.npy"
    test_prepare_p2_bs_samples(path=path)
    print("All tests passed.")
