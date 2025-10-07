import numpy as np
import matplotlib.pyplot as plt
import glob
from latqcdtools.statistics.jackknife import jackknife

def data_load():    
    file_list = sorted(glob.glob("./mass/*.dat"))
    real_data_all = []

    for fname in file_list:
        data = np.loadtxt(fname, comments='#')
        filtered = data[data[:, 3] == 0]
        real_values = filtered[:, 5]
        real_data_all.append(real_values)
    C_array = np.array(real_data_all)

    N_cfg = C_array.shape[0]
    flip_data = (C_array[:, :48] + np.flip(C_array[:, -48:])) / 2
    t = np.arange(flip_data.shape[1])
    print(f'{N_cfg} 个组态，{len(t)} 个时间点')

    return flip_data, t, N_cfg


def function(t, p, T):
    t_shifted = t - T/2
    y = np.zeros_like(t_shifted, dtype=float)
    max_index = -1
    for key in p.keys():
        if key.startswith('A') or key.startswith('m'):
            index = int(key[1:])
            max_index = max(max_index, index)
    for i in range(max_index + 1):
        A_key = f'A{i}'
        m_key = f'm{i}'
        if A_key in p and m_key in p:
            A = p[A_key]
            m = p[m_key]
            y += A * np.cosh(m * t_shifted)
    return y


if __name__ == "__main__":

    data, t, N_cfg = data_load()
    mean = np.mean(data, axis=0)

    p1 = {'m0': 0.65073937, 'A0': 1.0005e-15}
    p2 = {'m0': 0.629062, 'm1': 1.392983, 'A0': 2.264886e-15, 'A1': 1.809850e-30}

    y1 = function(t, p1, 96)
    y2 = function(t, p2, 96)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(t,np.abs(mean),marker='x')

    ax.semilogy(t, y1, color='#d62728', lw=2, ls='--', label=r'$N_s = 1$')
    ax.semilogy(t, y2, color='#1f77b4', lw=2, ls='--', label=r'$N_s = 2$')

    ax.set_xlabel("t")
    ax.set_ylabel("$G(t)$")
    ax.set_yscale('log')
    ax.set_ylim(1e-16, 2)
    ax.set_xlim(-1, 50)

    ax.tick_params(axis='both', direction='in', top=True, right=True)
    ax.legend(title="Number of states", loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()
