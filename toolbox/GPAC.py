import numpy as np

def GPAC(rY, k, j):
    # Initialize phi
    phi_j_kk = np.zeros((j, k))

    for jj in range(j):
        for kk in range(1, k+1):

            if kk == 1:
                phi_j_kk[jj, kk - 1] = rY[jj + 1] / rY[jj - kk + 1]

            else:
                shared_columns = np.zeros((kk, kk - 1))
                num_column = np.zeros((kk, 1))
                denom_column = np.zeros((kk, 1))

                for col in range(kk):
                    for row in range(kk):

                        if col != kk -1:
                            shared_columns[row, col] = rY[np.abs(jj + row - col)]

                        else:
                            num_column[row] = rY[np.abs(jj + row + 1)]
                            denom_column[row] = rY[np.abs(jj - kk + row + 1)]

                num = np.hstack((shared_columns, num_column))
                denom = np.hstack((shared_columns, denom_column))

                phi_j_kk[jj, kk - 1] = np.linalg.det(num) / np.linalg.det(denom)

    return phi_j_kk