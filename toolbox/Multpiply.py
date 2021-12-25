import numpy as np

#[[0, 2, 1, 0], [0, 1, 1, 3]]
# [AR order, Differencing, MA order, Seasonality]
# thata = [AR nonseasonal, AR seasonal, MA nonseasonal, MA seasonal]
def all_parts(lst, theta):
    param = 0 # for parameters in theta

    # for sublist in listoflist:
    na = lst[0]
    d = lst[1]
    nb = lst[2]
    s = lst[3]

    # Check if AR is activated
    if na != 0:
        ar_list = [1]

        # Check if AR is non-seasonal
        if s == 0:

            # Add AR non-seasonal
            for num in range(na):
                ar_list.append(theta[param])
                param += 1

        # Check if AR is seasonal
        else:

            # Add AR seasonal
            for num_seas in range(1, (na * s) + 1):

                # Check if we are at seasonality
                if num_seas % s == 0:
                    ar_list.append(theta[param])
                    param += 1

                else:
                    ar_list.append(0)

        # Get the final parameters of AR
        AR = np.poly1d(ar_list)

    else:
        AR = np.poly1d([1])



    # Check if MA is activated
    if nb != 0:
        ma_list = [1]

        # Check if MA is non-seasonal
        if s == 0:

            # Add MA non-seasonal
            for den in range(nb):
                ma_list.append(theta[param])
                param += 1

        # Check if MA is seasonal
        else:

            # Add MA seasonal
            for den_seas in range(1, (nb * s) + 1):

                # Check if we are at seasonality
                if den_seas % s == 0:
                    ma_list.append(theta[param])
                    param += 1

                else:
                    ma_list.append(0)

        # Get the final parameters of MA
        MA = np.poly1d(ma_list)


    else:
        MA = np.poly1d([1])

    # Check if Differencing is available
    if d != 0:
        if s == 0:
            diff = [1, -1]

        else:
            diff = [0] * (s + 1)
            diff[0] = 1
            diff[-1] = -1

        diff = np.poly1d(diff) ** d

    else:
        diff = np.poly1d([1])

    AR_with_diff = AR * diff

    return AR_with_diff, MA


