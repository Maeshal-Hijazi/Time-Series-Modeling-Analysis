# This function calculates the rolling mean and average for a 1-D dataframe such as Sales, AdBudget, or GDP
import numpy as np

def cal_rolling_mean_var(one_d_df):
    rolling_mean = []
    rolling_var = []
    for sample in range(one_d_df.shape[0]):
        rolling_mean.append(np.round(np.mean(one_d_df.iloc[0:(sample+1)]), 3))
        rolling_var.append(np.round(np.var(one_d_df.iloc[0:(sample+1)]), 3))

    return rolling_mean, rolling_var
