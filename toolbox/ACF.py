# ACF for a given dataset at a specific number of lag
import numpy as np
def autocorrelation(x, num_lags):
    if num_lags >= len(x):
        error = "ERROR: The number of lags used is higher than the data dimension"
        return print(error)

    else:
        ACF = [] # ACF list
        mean_x = np.mean(x)  # mean of x

        denom_sum = 0
        for t in range(len(x)):
            denom_sum += ((x[t] - mean_x) ** 2)

        for tau in range(num_lags + 1):
            num_sum = 0
            for t in range(tau, len(x)): # it is tau (not tau) because python indexing starts with 0
                  num_sum += ((x[t] - mean_x) * (x[t-tau] - mean_x))

            ACF.append(num_sum / denom_sum)

    ACF = np.concatenate((ACF[::-1][:-1], ACF))
    return ACF