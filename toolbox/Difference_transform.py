import numpy as np
import pandas as pd

def Difference_trans(data, factor):
    # Find the 1st order dataframe
    first_order_data = np.zeros((data.shape[0] - factor))

    for i in range(0, data.shape[0] - factor):
        first_order_data[i] = data.iloc[i + factor] - data.iloc[i]

    first_order_data = pd.Series(first_order_data)
    return first_order_data