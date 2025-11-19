import numpy as np
import glob
import os


def get_mean_err(data):
    mean = np.mean(data, axis=0)
    err = np.std(data, axis=0, ddof=1)
    return mean, err


def save_data(input_folder, pattern="*.npy"):
    files = sorted(glob.glob(os.path.join(input_folder, pattern)))

    for f in files:
        # 自动解析 p2
        basename = os.path.basename(f)  # temp_p2_3_bs.npy
        # 提取 3 -> int
        p2 = int(basename.split("_")[2])

        samples = np.load(f)  # shape (Nbs, T)

        mean = np.mean(samples, axis=0)
        err = np.std(samples, axis=0, ddof=1)

        data = np.column_stack([mean, err])  # shape (T, 2)
        header = "# mean err"
        np.savetxt(
            f"/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/p2_bs_mean_err/phi_p2_{p2}_mean_err.dat",
            data,
            header=header,
        )


if __name__ == "__main__":
    input_folder = "/home/george/Documents/WorkSpace/Lattice/corr_fit/data/processed/mom/bs_samples/"
    save_data(
        input_folder,
        pattern="phi_p2_*_bs.npy",
    )
    print("Done.")
