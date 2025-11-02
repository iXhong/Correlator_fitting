#home brewed jackknife & bootstrap
#@ixhong
#since 2025.8
import numpy as np
import glob

DEFAULTSEED = 7271978


def my_bootstrap(data,times):
    """
    my_bootstrap(data,times)
    Args:
        data:2d array,one configration per line
        times: times you want to bootstrap resample
    Return:
        mean,error:1d array
    """
    N,T = data.shape
    bs_samples = np.zeros((times,T))

    for i in range(times):
        #change the random method as in Analysistoolbox 
        rng = np.random.default_rng(DEFAULTSEED+i)
        chosen = rng.integers(0,N,N)
        bs_samples[i] = np.mean(data[chosen, :],axis=0)

    mean = np.mean(bs_samples, axis=0)
    # variance = np.sum((bs_samples - mean) ** 2, axis=0) / (times - 1 )
    # error = np.sqrt(variance)
    error = np.std(bs_samples, axis=0, ddof=1)

    return mean, error

# def my_bootstrap(data, times, seed=DEFAULTSEED):
#     """
#     Args:
#         data: 2d array, one configuration per line
#         times: bootstrap resample times
#         seed: int, base random seed for reproducibility
#     Return:
#         mean, error: 1d array
#     """
#     N, T = data.shape
#     bs_samples = np.zeros((times, T))

#     for i in range(times):
#         rng = np.random.default_rng(seed + i)  # 每次重新初始化 RNG，种子=seed+i
#         chosen = rng.integers(0, N, size=N)    # 抽取 N 个索引
#         bs_samples[i] = np.mean(data[chosen, :], axis=0)

#     mean = np.mean(bs_samples, axis=0)
#     variance = np.sum((bs_samples - mean) ** 2, axis=0) / (times - 1)
#     error = np.sqrt(variance)
#     return mean, error



def my_jackknife(data):
    """
    my_jackknife(data)
    Args:
        data:2d array
    Return:
        jk_mean,jk_error
    """
    N,T = data.shape
    jk_samples = np.zeros((N,T))

    for i in range(N):
        jk_samples[i] = np.mean(np.delete(data,i,axis=0),axis=0)
    
    jk_mean = np.mean(jk_samples,axis=0)
    jk_err = np.sqrt((N - 1) * np.mean((jk_samples - jk_mean[np.newaxis, :])**2, axis=0))

    return jk_mean,jk_err


def data_gen(N,T):
    #generate mock data
    """
    Args:
        N: number of configurations
        T: data number per conf
    Return:
        true_data: theoritecal data
        data: theoritecal data with noise added
        t: just self variable
    """
    T = 4
    m = 0.5  
    A = 1.0  
    N_cfg = N  
    noise_level = 0.8 

    t = np.linspace(0,5,T)

    true_data = A * (np.exp(-m * t))

    data = np.zeros((N_cfg, T))
    for i in range(N_cfg):
        noise = np.random.default_rng(i).normal(loc=0.0, scale=noise_level, size=T)
        data[i] = true_data + noise

    return true_data,data,t


def data_load(mu,path):
    """
    Function load data from two point correlation .dat file
    Args:
        mu(int): mumu ,0,1,2
        path: path to your correlator data,like:"./s3/mass/"
    Returns:
        C_array(np.array): 2d array,C data each line
        t(np.array): 1d time series 
    """ 
    
    file_list = sorted(glob.glob(f"{path}*.dat"))  # 根据你的文件名修改通配符

    real_data_all = []  # 用于存储每个文件的一维 real 数组

    for fname in file_list:
        data = np.loadtxt(fname, comments='#')
        
        filtered = data[data[:, 3] == mu]  # 第4列是 mumu
        
        real_values = filtered[:, 5]
        
        real_data_all.append(real_values)

    # 合并成二维数组，每一行是一个文件的数据
    C_array = np.array(real_data_all)
    t = np.array(range(C_array.shape[1]))

    return C_array, t


def jackknife_samples(data):
    """
    return jackknife samples
    """

    N,T = data.shape
    jk_samples = np.zeros((N,T))

    for i in range(N):
        jk_samples[i] = np.mean(np.delete(data,i,axis=0),axis=0)

    t_vals = np.array(range(T))
    return jk_samples,t_vals


#load ht's correlator data
def ref_data(path):
    #blocking
    def blocking(data,block_size):

        N = len(data)
        M = N//block_size
        blocked_data = np.zeros((M,block_size))
        for i in range(M):
            blocked_data[i,:] = np.array(data[i*block_size:(i+1)*block_size])

        return blocked_data

    data = np.loadtxt(path)
    data_fit = blocking(data[:,1],96)
    t_fit = np.array(range(len(data_fit[0])))
    return data_fit,t_fit

