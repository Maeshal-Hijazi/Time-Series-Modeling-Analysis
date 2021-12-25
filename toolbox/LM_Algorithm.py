import numpy as np
from scipy import signal
from numpy.linalg import inv, norm
from toolbox import Multpiply

# First, we need to write a function to simulate error
def simulate_e(output, theta, n_a, n_b, diff, seasonality):
    # n_b = len(theta) - n_a
    #
    # nmax = max(n_a, n_b)

    AR_all = np.poly1d([1])
    MA_all = np.poly1d([1])

    # Update the MA and AR parameters
    for i in range(len(n_a)):
        # if i == 0:
        AR, MA = Multpiply.all_parts([n_a[i], diff[i], n_b[i], seasonality[i]], theta[(sum(n_a[:i]) + sum(n_b[:i])) : (sum(n_a[:(i + 1)]) + sum(n_b[:(i + 1)]))])

        # else:
        #     AR, MA = Multpiply.all_parts([n_a[i], diff[i], n_b[i], seasonality[i]], theta[(sum(n_a[:i]) + sum(n_b[:i])) :
        #                                                                                   (sum(n_a[:(i + 1)]) + sum(n_b[:(i + 1)]))])

        AR_all = AR_all * AR
        MA_all = MA_all * MA


    ar_dlsim = list(AR_all.c)
    ma_dlsim = list(MA_all.c)


    # Make length of parameters the same
    if len(ar_dlsim) > len(ma_dlsim):
        ma_dlsim.extend([0] * (len(ar_dlsim) - len(ma_dlsim)))
    elif len(ar_dlsim) < len(ma_dlsim):
        ar_dlsim.extend([0] * (len(ma_dlsim) - len(ar_dlsim)))


    system = (ar_dlsim, ma_dlsim, 1) # sample is equal to 1 always
    _, error = signal.dlsim(system, output)

    return error


# Initialize common variable
delta = 10 ** (-6)
mu_max = 10 ** 21
epsilon = 0.001
mu = 0.01

# Step 1 of the LV Algorithm
def LM_step1(y, theta, n_a, n_b, diff, seasonality):

    # Get n (na + nb)
    n = len(theta)

    # Calculate error
    error = simulate_e(y, theta, n_a, n_b, diff, seasonality)

    # Calculate Sum Squared Error
    SSE_theta = error.T @ error

    # Initialize matrix X
    X = np.zeros((error.shape[0], n))

    # Loop through all the parameters
    for i in range(n):
        past_theta = theta[i]  # Save past theta value at current index
        theta[i] = theta[i] + delta  # Add the delta to the current index
        ei = simulate_e(y, theta, n_a, n_b, diff, seasonality)  # Error with the deviated delta for the current index

        # Calculate xi at current theta
        xi = (error - ei) / delta

        # Reimport the old value of theta at the index
        theta[i] = past_theta

        # Create matrix X
        X[:, i] = xi.T

    # Calculate A
    A = X.T @ X

    # Calculate gradient
    g = X.T @ error

    return SSE_theta, A, g

# Step 2 of the LV Algorithm
def LM_step2(y, theta, mu, n_a, n_b, diff, seasonality, A, g):

    # Get n (na + nb)
    n = len(theta)

    # Calculate delta theta (Change in theta)
    delta_theta = inv(A + mu * np.identity(n)) @ g
    delta_theta = [value[0] for value in delta_theta]

    # Get theta new and SSE new
    theta_new = [a + b for a, b in zip(theta, delta_theta)]
    error_new = simulate_e(y, theta_new, n_a, n_b, diff, seasonality)
    SSE_theta_new = error_new.T @ error_new

    return delta_theta, theta_new, SSE_theta_new, error_new

# Step 3 of the LV Algorithm
def LM_step3(y, mu, n_a, n_b, diff, seasonality):
    SSE = []

    # Initialize number of iterations
    iteration = 1

    n = sum(n_a) + sum(n_b)

    theta = []
    theta.extend([0] * n)

    # While loop to check the number of iterations
    while(iteration < 50):
        # Get variables from the output of the first step
        step1 = LM_step1(y, theta, n_a, n_b, diff, seasonality)
        SSE_theta = step1[0]
        A = step1[1]
        g = step1[2]

        # Get variables from the output of the second step
        step2 = LM_step2(y, theta, mu, n_a, n_b, diff, seasonality, A, g)
        delta_theta = step2[0]
        theta_new = step2[1]
        SSE_theta_new = step2[2]
        new_error = step2[3]

        if iteration == 1:
            SSE.append(SSE_theta)
        else:
            SSE.append(SSE_theta_new)


        # Checking if the error had decreased
        if SSE_theta_new < SSE_theta:

            # If what is decreased is very small
            if norm(delta_theta) < epsilon:

                # residual = new_error
                # y_adjusted = y - residual[:, 0]

                # Get the estimated theta
                theta_hat = theta_new

                # Calculate the variance of error
                var_e = SSE_theta_new / (y.shape[0] - len(theta))
                var_e = [value[0] for value in var_e]

                # Calculate the covariance of the estimated theta
                cov_theta_hat = var_e * inv(A)

                return theta_hat, cov_theta_hat, var_e, SSE
            # y_adjusted, residual

            # If not too small, continue decreasing
            else:

                # Save new theta
                theta = theta_new

                # Decrease mu
                mu = mu / 10

        # Checking if the error has increased
        else:
            # Increase mu
            mu = mu * 10

            # Check if mu became very large, then the algorithm has diverged
            if mu > mu_max:
                print('Algorithm is not converging.... mu > 10^21...... :(')
                break


        # Increment the number of iterations
        print(iteration)
        iteration += 1


        # Update theta
        # theta = theta_new