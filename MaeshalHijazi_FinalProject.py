# Final Project
# Professor Reza Jafari
# DATS 6450: Time series analysis and modeling
# 12/15/2021
# Maeshal Hijazi

################################################### Introduction #######################################################
'''
The data is the hourly load demand dataset for Texas from 01/01/2020 until 09/30/2021. There exist two excel files.
One file is for the 2020 data while the other is for the 2021 data. There exists several columns in the dataset other
than the time column. These other columns represent the hourly load demand for certain locations of Texas such as North,
South, East, West, etc. The last column is the total hourly load demand which represents the sum of all the locations.
The student will model the West region.
'''
################################################ Needed Libraries ######################################################

# Import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
from toolbox import ADF, KPSS, cal_rolling_mean_var, ACF, Difference_transform, GPAC, LM_Algorithm, Multpiply
import statsmodels.tsa.holtwinters as ets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.eval_measures import rmse
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from scipy.stats import chi2
from scipy.signal import tf2zpk


np.random.seed(123)
np.set_printoptions(suppress=True) # Set the exponential format to off


def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure(figsize=[20, 10])
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.grid()
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.grid()
    plt.show()

    return acf, pacf


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'



################################################# Data Loading #########################################################
print(color.BOLD + 'Part 1:' + color.END)
# Loading data
data_2020 = pd.read_excel('Native_Load_2020.xlsx')
data_2021 = pd.read_excel('Native_Load_2021.xlsx')

# Convert the two dataframes to arrays
data_2020_arr = np.array(data_2020)
data_2021_arr = np.array(data_2021)

# Concatenate the two arrays
data_arr = np.concatenate((data_2020_arr, data_2021_arr))

# Convert the concatenated array to a dataframe
data = pd.DataFrame(data_arr)

# Rename the column names to the original names
data.columns = data_2020.columns

# Rearrange Columns
data_num = data[['COAST', 'EAST', 'FWEST', 'NORTH', 'NCENT', 'SOUTH', 'SCENT', 'WEST', 'ERCOT']]

# Subtract mean from the data
data_num = data_num - data_num.mean()

# Normalize it
scaler = StandardScaler()
data_num_norm = scaler.fit_transform(data_num)

# Put the normalized data back to the dataframe
data[['COAST', 'EAST', 'FWEST', 'NORTH', 'NCENT', 'SOUTH', 'SCENT', 'WEST', 'ERCOT']] = data_num_norm

# Get the dependent variables
y = data['WEST']

X = data.drop(['WEST', 'HourEnding'], axis= 1).astype(float)


################################################# Description ##########################################################
# Part a: Pre-processing dataset (Dataset cleaning for missing observation)
print(color.BOLD + 'Description_Part a:' + color.END)
print(data.isnull().sum())
print(f"\nThe number of missing data is {data.isnull().sum().sum()}")


# Part b: Plot of the dependent variable versus time.
print(color.BOLD + '\nDescription_Part b:' + color.END)

# Get the xlabels and xlableticks
xlabeltics = [0, 30, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365, 396, 424, 455, 485, 516, 546, 577, 608, 638]\
             * np.array([24])
xlabels = ['Jan-20', 'Feb-20', 'Mar-20', 'Apr-20', 'May-20', 'Jun-20', 'Jul-20', 'Aug-20', 'Sep-20', 'Oct-20', 'Nov-20'
           , 'Dec-20', 'Jan-21', 'Feb-21', 'Mar-21', 'Apr-21', 'May-21', 'Jun-21', 'Jul-21', 'Aug-21', 'Sep-21']

# Creating the plot for the dependent variable
plt.figure(figsize= [20, 10])
plt.plot(y)
plt.xticks(xlabeltics, xlabels, fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.title('Hourly Load Demand for West Region in Texas',  fontsize= 20)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Load Demand', fontsize= 15)
plt.grid()
plt.savefig('Plot Dependent Variable.png')
plt.show()


# Part c: ACF/PACF of the dependent variable.
print(color.BOLD + '\nDescription_Part c:' + color.END)

# Calculate ACF of dependent variable for 90 lags
y_acf = ACF.autocorrelation(y, num_lags= 90)

# Recreate ACF plot x- axis
x_axis = np.arange(-90, 91)

# Creating the plot for the ACF of the dependent variable
plt.figure(figsize= [20, 10])
(marker, stemlines, baselines)= plt.stem(x_axis, y_acf, use_line_collection= True, markerfmt='o')
plt.setp(marker, color = 'cornflowerblue', marker='o')
plt.setp(baselines, color='cornflowerblue', linewidth=2, linestyle='-' )
plt.setp(stemlines, color = 'black', linestyle = '-')
m = 1.96 /np.sqrt(y.shape[0])
plt.axhspan(-m,m, alpha = .2, color='lightskyblue')
plt.title('ACF for Hourly Load Demand for West Region in Texas',  fontsize= 20)
plt.xlabel('#Lags', fontsize = 15)
plt.ylabel('ACF Value', fontsize = 15)
plt.xticks(fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.grid()
plt.savefig('ACF Dependent')
plt.show()

# Creating the plot for the PACF of the dependent variable
fig, ax = plt.subplots(1, 1, figsize = [15, 10])
statsmodels.graphics.tsaplots.plot_pacf(y, lags = 90, method= 'ywm', ax = ax)
plt.title('PACF for Hourly Load Demand for West Region in Texas', fontsize = 15)
plt.xlabel('#Lags', fontsize = 15)
plt.ylabel('PACF Value', fontsize = 15)
plt.xticks(fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.grid()
plt.savefig('PACF Dependent')
plt.show()


# Part d: Correlation Matrix with seaborn heatmap with the Pearson’s correlation coefficient.
print(color.BOLD + '\nDescription_Part d:' + color.END)

# Creating the plot for the Heatmap of the correlation matrix of the dependent variable
plt.figure(figsize= [10, 10])
sns.heatmap(data.drop(['HourEnding'], axis = 1).astype(float).corr(),
            vmin = -1,
            vmax = 1,
            center = 0,
            square = True,
            annot = data.drop(['HourEnding'], axis = 1).astype(float).corr())
plt.yticks(rotation = 90)
plt.title('Heatmap of the Correlation Matrix', fontsize= 20)
plt.savefig('Heatmap of the correlation matrix.png')
plt.show()


# Part e: Split the dataset into train set (80%) and test set (20%).
print(color.BOLD + '\nDescription_Part e:' + color.END)


x_train, x_test, y_train, y_test = train_test_split(X, y, shuffle = False, test_size=0.2) # Splitting Input and Output to testing and
                                                                         # training

################################################# Stationarity #########################################################
print(color.BOLD + '\nStationarity:' + color.END)

# Check stationarity by ADF
print('The ADF result of the dependent variable')
ADF.ADF_Cal(y)

# Check stationarity by KPSS
print('\nThe KPSS result of the dependent variable')
KPSS.kpss_test(y)

# Check stationarity by rolling mean and variance
rolling_y = cal_rolling_mean_var.cal_rolling_mean_var(y)
rolling_mean_y = rolling_y[0]
rolling_var_y = rolling_y[1]

# Creating the plot for the rolling mean of the dependent variable
plt.figure(figsize= [20, 10])
plt.plot(rolling_mean_y)
plt.title('Rolling Mean for Hourly Load Demand for West Region in Texas',  fontsize= 20)
plt.xticks(xlabeltics, xlabels, fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Rolling Mean of Load Demand', fontsize= 15)
plt.grid()
plt.savefig('Rolling Mean')
plt.show()

# Creating the plot for the rolling variance of the dependent variable
plt.figure(figsize= [20, 10])
plt.plot(rolling_var_y)
plt.title('Rolling Variance for Hourly Load Demand for West Region in Texas',  fontsize= 20)
plt.xticks(xlabeltics, xlabels, fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Rolling Variance of Load Demand', fontsize= 15)
plt.grid()
plt.savefig('Rolling Variance')
plt.show()

print('\nThe Rolling mean and variance with the ADF and KPSS indicate that the depednent variable is not stationary\n')

##### Convert to Stationary (Mean stabilized)

y_diff_trans = Difference_transform.Difference_trans(y, 24)

# Check stationarity by ADF of the transformed data
print('The ADF result of the transformed dependent variable')
ADF.ADF_Cal(y_diff_trans)

# Check stationarity by KPSS of the transformed data
print('\nThe KPSS result of the transformed dependent variable')
KPSS.kpss_test(y_diff_trans)

# Check stationarity of the  by rolling mean and variance
rolling_y = cal_rolling_mean_var.cal_rolling_mean_var(y_diff_trans)
rolling_mean_y_transformed = rolling_y[0] # rolling mean of the
rolling_var_y_transformed = rolling_y[1]

# Creating the plot for the rolling mean of one time difference transformation of the dependent variable
plt.figure(figsize= [20, 10])
plt.plot(rolling_mean_y_transformed)
plt.title('Rolling Mean for Hourly Load Demand for West Region in Texas (Transformed)',  fontsize= 20)
plt.xticks(xlabeltics, xlabels, fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Rolling Mean of Load Demand', fontsize= 15)
plt.grid()
plt.savefig('Rolling Mean Transformed')
plt.show()

# Creating the plot for the rolling variance of one time difference transformation of the dependent variable
plt.figure(figsize= [20, 10])
plt.plot(rolling_var_y_transformed)
plt.title('Rolling Variance for Hourly Load Demand for West Region in Texas (Transformed)',  fontsize= 20)
plt.xticks(xlabeltics, xlabels, fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Rolling Variance of Load Demand', fontsize= 15)
plt.grid()
plt.savefig('Rolling Variance Transformed')
plt.show()


print('\nThe Rolling mean and variance have stabilized after the first transformation of the dependent variable \n')

# Calculate ACF of the stationarized dependent variable for 90 lags
y_diff_trans = Difference_transform.Difference_trans(y, 24)
y_acf_diff = ACF.autocorrelation(y_diff_trans, num_lags= 90)

# Creating the plot for the ACF of the dependent variable
plt.figure(figsize= [20, 10])
(marker, stemlines, baselines)= plt.stem(x_axis, y_acf_diff, use_line_collection= True, markerfmt='o')
plt.setp(marker, color = 'cornflowerblue', marker='o')
plt.setp(baselines, color='cornflowerblue', linewidth=2, linestyle='-' )
plt.setp(stemlines, color = 'black', linestyle = '-')
m = 1.96 /np.sqrt(y_diff_trans.shape[0])
plt.axhspan(-m,m, alpha = .2, color='lightskyblue')
plt.title('ACF for Hourly Load Demand for West Region in Texas (After Difference Transformations)',  fontsize= 20)
plt.xlabel('#Lags', fontsize = 15)
plt.ylabel('ACF Value', fontsize = 15)
plt.xticks(fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.grid()
plt.savefig('ACF Dependent after stationarized')
plt.show()

# Creating the plot for the PACF of the dependent variable
fig, ax = plt.subplots(1, 1, figsize = [15, 10])
statsmodels.graphics.tsaplots.plot_pacf(y_diff_trans, lags = 90, method= 'ywm', ax = ax)
plt.title('PACF for Hourly Load Demand for West Region in Texas (After Difference Transformations)', fontsize = 15)
plt.xlabel('#Lags', fontsize = 15)
plt.ylabel('PACF Value', fontsize = 15)
plt.xticks(fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.grid()
plt.savefig('PACF Dependent after stationarized')
plt.show()



########################################### Time series Decomposition ##################################################
print(color.BOLD + '\nTime series Decomposition:' + color.END)

res = STL(y, period= 24).fit() # Get the decomposition

# Plot the original, trend, seasonality, and residual for the dependent variable
fig, ax = plt.subplots(4, 1, figsize=[20, 10])
ax[0].set_title('STL Decomposition')
ax[0].plot(y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Original')
ax[0].grid()
ax[0].set_xticks(xlabeltics)
ax[0].set_xticklabels(xlabels)

ax[1].plot(res.trend)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Trend')
ax[1].grid()
ax[1].set_xticks(xlabeltics)
ax[1].set_xticklabels(xlabels)

ax[2].plot(res.seasonal)
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Seasonal')
ax[2].grid()
ax[2].set_xticks(xlabeltics)
ax[2].set_xticklabels(xlabels)

ax[3].plot(res.resid)
ax[3].set_xlabel('Time')
ax[3].set_ylabel('Residual')
ax[3].grid()
ax[3].set_xticks(xlabeltics)
ax[3].set_xticklabels(xlabels)

plt.savefig('Decomposition.png')
plt.show()

# Calculate the trend strength and see if it higher than zero
strength_trend = 1 - (np.var(res.resid)/(np.var(res.trend + res.resid)))

if strength_trend < 0:
    strength_trend = 0

print('The strength of trend for the dependent variable is', strength_trend)

# Calculate the seasonality strength and see if it higher than zero
strength_seasonality = 1 - (np.var(res.resid)/(np.var(res.seasonal + res.resid)))

if strength_seasonality < 0:
    strength_seasonality = 0

print('The strength of seasonality for the dependent variable is', strength_seasonality)

# Calculate detrended data and plot it
detrended = y / res.trend

plt.figure(figsize=[20, 10])
plt.plot(detrended)
plt.title('Detrended Dependent Variable (Multiplicative)', fontsize= 20)
plt.yticks(fontsize= 12.5)
plt.xticks(xlabeltics, xlabels, fontsize= 12.5)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Detrended Dependent Variable', fontsize= 15)
plt.grid()
plt.savefig('Detrended')
plt.show()

# Calculate seasonally adjusted data and plot it
seasonally_adjusted = y / res.seasonal

plt.figure(figsize=[20,10])
plt.plot(seasonally_adjusted)
plt.title('Seasonally Adjusted Dependent Variable (Multiplicative)', fontsize= 20)
plt.yticks(fontsize= 12.5)
plt.xticks(xlabeltics, xlabels, fontsize= 12.5)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Seasonally Adjusted Dependent Variable', fontsize= 15)
plt.grid()
plt.savefig('Seasonally Adjusted')
plt.show()



############################################### Holt-Winters method ####################################################
print(color.BOLD + '\nHolt-Winters method:' + color.END)
y_train, y_test = np.array(y_train), np.array(y_test) # Convert train and test

y_pred_hl = [np.NAN, np.NAN] # Initialize y pred with 0 at
error_pred_hl = [np.NAN, np.NAN] # Initialize error for the prediction
error_pred_sqrd_hl = [np.NAN, np.NAN] # Calculate error squared

print('Training started ...')
# Get the forecast of y_test using Holt Linear Method
holtt= ets.ExponentialSmoothing(y_train, trend='add', damped_trend=True, seasonal='add', seasonal_periods= 24).fit(
smoothing_level = 0.1, smoothing_seasonal = 0.9, smoothing_trend = 0.02)

print('Testing started ...')
y_cast_hl = holtt.forecast(steps = len(y_test))

error_cast_hl = [] # Initialize error for forecast
error_cast_sqrd_hl = [] # Calculate error squared
for forecast in range(len(y_test)):
    error_cast_hl.append(y_test[forecast] - y_cast_hl[forecast])
    error_cast_sqrd_hl.append(error_cast_hl[forecast] ** 2)

# Get the xlableticks for the forecast of the test set
test_index = []
for i in range(len(y_train)+1, len(y)+1):
    test_index.append(i)

# Plotting the Holt's Winter forecast of the test set
plt.figure(figsize=[20, 10])
plt.plot(test_index, y_test)
plt.plot(test_index, y_cast_hl)
plt.xticks(xlabeltics[17:], xlabels[17:], fontsize= 12.5)
plt.title("Holt's Winter Forecast of the Test Set",  fontsize= 20)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Load Demand', fontsize= 15)
plt.grid()
plt.legend(['Test Set', "Holt's Winter Forecast"])
plt.savefig('Holt Winter Forecast')
plt.show()

print(f"MSE for Holt's Winter Method is: {np.nanmean(error_cast_sqrd_hl)}")



################################################ Feature selection #####################################################
print(color.BOLD + '\nFeature selection:' + color.END)
print(color.UNDERLINE + '\nFeature selection_SVD_Before PCA:' + color.END)

# Construct H
H = X.T @ X # This array is 8 x 8 matrix

# Calculate the SVD
s, d, v = np.linalg.svd(H)
d = ["{:0.4f}".format(dd) for dd in d] # Converting to to float
print("Singular Values =\n", list(map(float, d)))


print(color.UNDERLINE + '\nFeature selection_Conditional Number_Before PCA :' + color.END)
# Calculate the conditional number
k = np.linalg.cond(X)

print(f"The conditional number for the features is: {k}")
print('\nBased on the Singular values and Conditional number, moderate collinearity exists and some features has to be removed.')


print(color.UNDERLINE + '\nFeature selection_PCA:' + color.END)
# See how many features we need using PCA
pca = PCA(n_components = "mle", svd_solver= 'full')
pca.fit(X)
# X_pca = pca.transform(X)
plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_)) + 1, 1), np.cumsum(pca.explained_variance_ratio_))
plt.title('ROC of variance')
plt.ylabel('Variance Percentage')
plt.xlabel('Feature Number')
plt.grid()
plt.savefig('ROC PCA')
plt.show()

print('From ROC plot, we can see only 3 features are needed')

# Calculate PCS for 3 features only
pca = PCA(n_components = 3)
pca.fit(X)
X_pca = pca.transform(X)


print(color.UNDERLINE + '\nFeature selection_SVD_After PCA:' + color.END)

# Construct H
H_pca = X_pca.T @ X_pca # This array is 8 x 8 matrix

# Calculate the SVD
s, d, v = np.linalg.svd(H_pca)
d = ["{:0.4f}".format(dd) for dd in d] # Converting to to float
print("Singular Values after PCA =\n", list(map(float, d)))


print(color.UNDERLINE + '\nFeature selection_Conditional Number_After PCA :' + color.END)
# Calculate the conditional number
k = np.linalg.cond(X_pca)

print(f"The conditional number for the features after PCA is: {k}")




print(color.UNDERLINE + '\nFeature selection_Backward Stepwise Regression:' + color.END)
print('This part will be done in the' + color.BOLD + ' Multiple Linear Regression ' + color.END + 'model')



################################################### Base-Models ########################################################
print(color.BOLD + '\nBase Models:' + color.END)

# Average Method
print(color.UNDERLINE + '\nBase Models_Average:' + color.END)

y_cast_avg = [] # Initialize y forecasted
error_cast_avg = [] # Initialize error for forecast
error_cast_sqrd_avg = [] # Calculate error squared

# Calculating the h-step ahead forecast by the averaging method for each y_t
for forecast in range(len(y_test)):
    y_cast_avg.append(np.mean(y_train))
    error_cast_avg.append(y_test[forecast] - y_cast_avg[forecast])
    error_cast_sqrd_avg.append(error_cast_avg[forecast] ** 2)

print(f"MSE for Average Method is: {np.mean(error_cast_sqrd_avg)}")


# Naive Method
print(color.UNDERLINE + '\nBase Models_Naive:' + color.END)

y_cast_naive = [] # Initialize y forecasted
error_cast_naive = [] # Initialize error for forecast
error_cast_sqrd_naive = [] # Calculate error squared

# Calculating the h-step ahead forecast by the naive method for each y_t
for forecast in range(len(y_test)):
    y_cast_naive.append(y_train[-1])
    error_cast_naive.append(y_test[forecast] - y_cast_naive[forecast])
    error_cast_sqrd_naive.append(error_cast_naive[forecast] ** 2)

print(f"MSE for Naive Method is: {np.mean(error_cast_sqrd_naive)}")


# Drift Method
print(color.UNDERLINE + '\nBase Models_Drift:' + color.END)

y_cast_drift = [] # Initialize y forecasted
error_cast_drift = [] # Initialize error for forecast
error_cast_sqrd_drift = [] # Calculate error squared

# Calculating the h-step ahead forecast by the drift method for each y_t
for forecast in range(len(y_test)):
    h = forecast + 1 # h time-step
    y_cast_drift.append(y_train[-1] + (h * (y_train[-1] - y_train[0]) / (len(y_train) - 1)))
    error_cast_drift.append(y_test[forecast] - y_cast_drift[forecast])
    error_cast_sqrd_drift.append(error_cast_drift[forecast] ** 2)

print(f"MSE for Drift Method is: {np.mean(error_cast_sqrd_drift)}")


# SES Method
print(color.UNDERLINE + '\nBase Models_SES:' + color.END)

# Initialize coefficients for SES (Assuming alpha is 0.5)
alpha = 0.5
l0 = y_train[0]

y_cast_ses = [] # Initialize y forecasted
error_cast_ses = [] # Initialize error for forecast
error_cast_sqrd_ses = [] # Calculate error squared

# Calculating the h-step ahead forecast by the SES method for each y_t
total = 0
TT = len(y_train)
for j in range(TT):
    total += (alpha * ((1 - alpha) ** j) * y_train[TT - 1 - j])

total += (((1 - alpha) ** (TT)) * l0)

for forecast in range(len(y_test)):
    y_cast_ses.append(total)
    error_cast_ses.append(y_test[forecast] - y_cast_ses[forecast])
    error_cast_sqrd_ses.append(error_cast_ses[forecast] ** 2)

print(f"MSE for SES Method is: {np.mean(error_cast_sqrd_ses)}")

# Plotting All base methods
plt.figure(figsize= [20, 10])
plt.plot(test_index, y_test)
plt.plot(test_index, y_cast_avg, linewidth = 3)
plt.plot(test_index, y_cast_naive, linewidth = 3)
plt.plot(test_index, y_cast_drift, linewidth = 3)
plt.plot(test_index, y_cast_ses, linewidth = 3)
plt.xticks(xlabeltics[17:], xlabels[17:], fontsize= 12.5)
plt.title('Base Models Forecast',  fontsize= 20)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Load Demand', fontsize= 15)
plt.grid()
plt.legend(['Test Set', 'Average Method', 'Naive Method', 'Drift Method', 'SES Method'])
plt.savefig('Base Models.png')
plt.show()



########################################### Multiple Linear Regression #################################################
# Part a: You need to include the complete regression analysis into your report. Perform one-step ahead prediction and
#         compare the performance versus the test set.
print(color.BOLD + '\nMultiple Linear Regression:' + color.END)
print(color.UNDERLINE + '\nMultiple Linear Regression_Part a:' + color.END)

# Add a column of ones to x
one = np.ones((len(X), 1))

# Converting the column of ones to a dataframe
one = pd.DataFrame(one, columns = ['const'])

# Add the column of ones to the features
X_LR = pd.concat((one, X), axis = 1)


# Get training and testing dataframes for input and output for regression model
x_train_LR, x_test_LR, y_train_LR, y_test_LR = \
    train_test_split(X_LR, y, shuffle= False, test_size=0.2) # Splitting Input and Output dataframes to testing
                                                             # and training

# Create model
model = sm.OLS(y_train_LR, x_train_LR).fit()

# Print model summary
print(model.summary())

# Function to test Regression model
def forecast_LR(x_test_LR):
    y_cast = model.predict(x_test_LR)

    e_LR = []
    SE_LR = []
    for k in range(len(y_test_LR)):
        e_LR.append((np.array(y_test_LR)[k] - np.array(y_cast)[k]))
        SE_LR.append(e_LR[k] ** 2)

    # Calculate MSE for Linear Regression
    MSE_LR = np.mean(SE_LR)
    print(f"MSE for Multiple Linear Regression Method is: {MSE_LR}")

    return SE_LR, MSE_LR, y_cast

print('\nThe MSE for all features')
# Forecast for all features are included
SE_LR, MSE_LR, y_cast = forecast_LR(x_test_LR)

# Since the mean was subtracted from the the feature matrix, remove the constant term
x_train_LR = x_train_LR.drop(['const'], axis= 1)
x_test_LR = x_test_LR.drop(['const'], axis= 1)

# Create model
model = sm.OLS(y_train_LR, x_train_LR).fit()

# Print model summary
print(model.summary())

print('\nThe MSE for all features after removing the constant term')

# Forecast after removing the constant term
SE_LR, MSE_LR, y_cast = forecast_LR(x_test_LR)


# Function that calculates the VIF for the given feature
def VIF(x_train_LR):
    # Create dataframe for VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = x_train_LR.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(x_train_LR.values, i)
                       for i in range(len(x_train_LR.columns))]

    return vif_data

# Functions to drop the feature with high VIF
def drop_high_VIF(x_train_LR):
    # Get the VIF of the remained features
    vif_data = VIF(x_train_LR)

    # Get the index of the maximum VIF
    index = vif_data.iloc[:, 1].idxmax()
    feature = vif_data.iloc[index, 0]

    # Drop the feature with the highest VIF
    x_train_LR = x_train_LR.drop([feature], axis = 1)

    # Get the VIF of the remained features after the removed
    vif_data = VIF(x_train_LR)

    return x_train_LR, vif_data

# For loop to get the three most uncorrelated features
for feature in range(x_train_LR.shape[1], 3, -1):
    x_train_LR, Vif = drop_high_VIF(x_train_LR)
    model = sm.OLS(y_train_LR, x_train_LR).fit()


    s = []
    for i in range(Vif.shape[0]):
        s.append(Vif.iloc[i, 0])

    x_test_LR = x_test_LR[s]


print('\nThe MSE for the remaining 3 features')
# Forecast for the remaining three methods
SE_LR, MSE_LR, y_cast = forecast_LR(x_test_LR)

# Print summary of the reduced model
print(model.summary())

# Plot y test and y pred
plt.figure(figsize=[20, 10])
plt.plot(test_index, y_test_LR)
plt.plot(test_index, y_cast)
plt.xticks(xlabeltics[17:], xlabels[17:], fontsize= 12.5)
plt.title("Multiple Linear Regression Forecast of the Test Set",  fontsize= 20)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Load Demand', fontsize= 15)
plt.legend(['Test Set', "Regression Forecast"])
plt.grid()
plt.savefig('Regression Forecast')
plt.show()

# # Convert the tests to arrays
# output_test = np.array(y_test_LR)
# y_cast = np.array(y_cast)

# Predict the x test
y_one_step = model.predict(x_train_LR)

# Convert the tests to arrays
y_train_LR = np.array(y_train_LR)
y_one_step = np.array(y_one_step)

# Initialize list for the residual error (training)
residual_LR = []
for k in range(len(y_train_LR)):
    residual_LR.append((y_train_LR[k] - y_one_step[k]))

# Plot y train and one step prediction
plt.figure(figsize=[20, 10])
plt.plot(y_train_LR)
plt.plot(y_one_step)
plt.xticks(xlabeltics[:17], xlabels[:17], fontsize= 12.5)
plt.title("Multiple Linear Regression One Step Ahead Prediction of the Train Set",  fontsize= 20)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Load Demand', fontsize= 15)
plt.legend(['Train Set', "Regression One-Step Lookahead"])
plt.grid()
plt.savefig('Regression One-Step Lookahead')
plt.show()


# Part b:  Hypothesis tests analysis: F-test, t-test.
print(color.UNDERLINE + '\nMultiple Linear Regression_Part b:' + color.END)
print('Based on the model summary, the t-test and the F-test were passed')


# Part c: AIC, BIC, RMSE, R-squared and Adjusted R-squared
print(color.UNDERLINE + '\nMultiple Linear Regression_Part c:' + color.END)

print(f'AIC is: {model.aic}')
print(f'BIC is: {model.bic}')
print(f'Root Mean Squared Error (RMSE) is: {rmse(y_test, y_cast)}')
print(f'R-squared is: {model.rsquared}')
print(f'Adjusted R-squared is: {model.rsquared_adj}')


# Part d: ACF of residuals
print(color.UNDERLINE + '\nMultiple Linear Regression_Part d:' + color.END)

# Get the ACF of the error of the regression Model
residual_LR_acf = ACF.autocorrelation(residual_LR, num_lags= 90)

# Creating the plot for the ACF of the error of the regression model (ON TRAINING)
plt.figure(figsize= [20, 10])
(marker, stemlines, baselines)= plt.stem(x_axis, residual_LR_acf, use_line_collection= True, markerfmt='o')
plt.setp(marker, color = 'cornflowerblue', marker='o')
plt.setp(baselines, color='cornflowerblue', linewidth=2, linestyle='-' )
plt.setp(stemlines, color = 'black', linestyle = '-')
m = 1.96 /np.sqrt(len(residual_LR))
plt.axhspan(-m,m, alpha = .2, color='lightskyblue')
plt.title('ACF for the Residual of the Multiple Linear Regression',  fontsize= 20)
plt.xlabel('#Lags', fontsize = 15)
plt.ylabel('ACF Value', fontsize = 15)
plt.xticks(fontsize= 12.5)
plt.yticks(fontsize= 12.5)
plt.grid()
plt.savefig('ACF Residual of Regression')
plt.show()


# Part e: Q-value
print(color.UNDERLINE + '\nMultiple Linear Regression_Part e:' + color.END)

# Calculating Q
Q_LR = len(y) * np.sum(np.square(residual_LR_acf[(90+1):]))
print(f"The Q value of the residual is {Q_LR}")


# Part f: Variance and mean of the residuals.
print(color.UNDERLINE + '\nMultiple Linear Regression_Part f:' + color.END)

# Calculating mean and variance of the residual
mean_e_LR = np.mean(residual_LR)
var_e_LR = np.var(residual_LR)

print(f"The mean of the residual is {mean_e_LR}")
print(f"The variance of the residual is {var_e_LR}")


######################################## ARMA and ARIMA and SARIMA model ###############################################
# Part a: Preliminary model development procedures and results. (ARMA model order determination). Pick at least two
#         orders using GPAC table.
print(color.BOLD + '\nGPAC & ACF/PACF:' + color.END)

# Chi-Square function
def CHi_Square(resid, n_a, n_b, lags):
    alpha = 0.01 # Threshold

    # Calculate Q
    Q = sm.stats.acorr_ljungbox(resid, lags=[lags])[0][0]

    # Determine the degree of freedom
    DOF = lags - n_a - n_b

    # Calculate Qc
    chi_critical = chi2.ppf(1 - alpha, DOF)

    # CHeck if residual is white or not
    if Q < chi_critical:
        print('The residual is white :)')

    else:
        print('The residual is NOT white :(')

    return Q, chi_critical

# Perform differencing on the dataset
y_diff1 = Difference_transform.Difference_trans(y, 1)

y_diff2 = Difference_transform.Difference_trans(y_diff1, 24)

y_diff3 = Difference_transform.Difference_trans(y_diff2, 1)

y_diff4 = Difference_transform.Difference_trans(y_diff3, 168)

# Get the ACF and PACF of the last difference
y_acf, y_pacf = ACF_PACF_Plot(y_diff4, 175)

# Caclulate the GPAC matrix of the differenced dependent variable
G_Table1 = GPAC.GPAC(y_acf, k = 26, j = 26)

# Updating the xticks and labels
xticklabels = np.arange(1, 27)

# Creating the plot for the GPAC Table
plt.figure(figsize= [16, 16])
sns.heatmap(G_Table1,
            vmin = -1,
            vmax = 1,
            center = 0,
            square = True,
            fmt='.2f',
            xticklabels = xticklabels,
            annot = True)
plt.title('GPAC Table for dependent variable after differencing')
plt.savefig('GPAC')
plt.show()

print('\nEstimated order (non-seasonal) from the GPAC is na = 24 and nb = 0')


######################################## LM Algorithm ###############################################
# Display the parameter estimates, the standard deviation of the parameter estimates and confidence intervals.
print(color.BOLD + '\nLM Algorithm:' + color.END)

print('\nSince the data is highly seasonal and complex, a multiplicative model was need')

# Three models      (24,2,0) x (0,1,1,24) x (1,1,2,168)
na = [24, 0, 1]
nb = [0, 1, 2]
diff = [2, 1, 1]
seasonal = [0, 24, 168]

# Get the approximated parameters, the covariance matrix, the varience of the error and the SSE of the multiplicative model
theta_hat, cov_theta_hat, var_e, SSerror = LM_Algorithm.LM_step3(y, 0.01, na, nb, diff, seasonal)

print(f"\nThe estimated parameters are:\n{theta_hat}")


############################################## Diagnostic Analysis #####################################################
# Part a: confidence intervals, zero/pole cancellation, chi-square test
print(color.BOLD + '\nDiagnostic Analysis:' + color.END)
print(color.UNDERLINE + '\nDiagnostic Analysis_Part a:' + color.END)

# Function calculates the zeros and poles of a submodel
def Zeros_Poles(theta_hat, n_a):
    # Get the parameters for AR and MA process
    param_na = theta_hat[:n_a]
    param_nb = theta_hat[n_a:]

    # Make length of parameters the same
    if len(param_na) > len(param_nb):
        param_nb.extend([0] * (len(param_na) - len(param_nb)))
    elif len(param_na) < len(param_nb):
        param_na.extend([0] * (len(param_nb) - len(param_na)))

    ar_dlsim = np.r_[1, param_na]
    ma_dlsim = np.r_[1, param_nb]

    # Zeros and Poles
    zeros, poles, _ = tf2zpk(ma_dlsim, ar_dlsim)
    for z in range(len(zeros)):
        if zeros[z] != 0:
            print(f'Zero # {z + 1} is: {zeros[z]}')

    for p in range(len(poles)):
        if poles[p] != 0:
            print(f'Pole # {p + 1} is: {poles[p]}')

# Confidence Interval
def conf_interval(n_a, n_b, theta, cov_theta_hat):
    # Loop through AR process
    for a in range(n_a):
        print(f'{theta[a] - (2 * np.sqrt(cov_theta_hat[a][a]))} < a{a+1} < {theta[a] + (2 * np.sqrt(cov_theta_hat[a][a]))}')

    # Loop through MA process
    for b in range(n_b):
        print(
            f'{theta[n_a + b] - (2 * np.sqrt(cov_theta_hat[n_a + b][n_a + b]))} < b{b + 1} < '
            f'{theta[n_a + b] + (2 * np.sqrt(cov_theta_hat[n_a + b][n_a + b]))}')

# Initialize AR and MA model
AR_all = np.poly1d([1])
MA_all = np.poly1d([1])

# Get the multiplicative model
for i in range(len(na)):

    low = (sum(na[:i]) + sum(nb[:i]))
    high = (sum(na[:(i + 1)]) + sum(nb[:(i + 1)]))

    theta = theta_hat[(sum(na[:i]) + sum(nb[:i])) : (sum(na[:(i + 1)]) + sum(nb[:(i + 1)]))]

    AR, MA = Multpiply.all_parts([na[i], diff[i], nb[i], seasonal[i]], theta)

    # Get the confidence intervals for the sub models
    conf_interval(na[i], nb[i], theta, cov_theta_hat[low : high, low : high])

    # Get the zeros and poles for the submodel
    Zeros_Poles(theta, na[i])
    print()

    # Update AR and MA for the full model
    AR_all = AR_all * AR
    MA_all = MA_all * MA

# Initialize the list for the one-step ahead predicition
y_one = []

y_AR_one = []
y_MA_one = []

# Initialize step as 1
step = 1

# One step ahead prediction
for t in range(len(y_train)):
    yt = 0

    for param_ar_index in range(step, len(AR_all.c)):

        if AR_all.c[param_ar_index] != 0:

            if (t - param_ar_index + step) >= 0:
                yt += AR_all.c[param_ar_index] * y[t - param_ar_index + step]

    y_AR_one.append(-1 * yt)

    et = 0

    # Error values before 'step' are plus t such as (e(t+1), e(t+2), ......), then they are zeros
    for param_ma_index in range(step, len(MA_all.c)):

        if MA_all.c[param_ma_index] != 0:

            if (t - param_ma_index + step) >= 0:
                if (t - param_ma_index) >= 0:
                    et += MA_all.c[param_ma_index] * (y[t - param_ma_index + step] - y_one[t - param_ma_index])

                else:
                    et += MA_all.c[param_ma_index] * y[t - param_ma_index + step]

    y_MA_one.append(et)

    y_one.append(y_AR_one[t] + y_MA_one[t])

# Plot the train vs the one step ahead prediction
plt.figure(figsize=[20, 10])
plt.plot(np.array(y_train)[1:])
plt.plot(y_one[:-1])
plt.xticks(xlabeltics[:17], xlabels[:17], fontsize= 12.5)
plt.title("One-step Ahead Prediction for Multiplicative Model",  fontsize= 20)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Load Demand', fontsize= 15)
plt.legend(['Train Set', "One-step Ahead"])
plt.grid()
plt.savefig('Multiplicative One Step Ahead')
plt.show()

# Plot the first 250 samples from train vs the one step ahead prediction
plt.figure(figsize=[8,8])
plt.plot(np.array(y_train[1:251]))
plt.plot(y_one[:250])
plt.xticks(xlabeltics[:1], xlabels[:1])
plt.title("One-step Ahead Prediction for Multiplicative Model (First 250 samples)")
plt.xlabel('Time (hr)')
plt.ylabel('Load Demand')
plt.legend(['Train Set', "One-step Ahead"])
plt.grid()
plt.savefig('Multiplicative One Step Ahead (First 250 samples)')
plt.show()

# Plot the last 250 samples from train vs the one step ahead prediction
plt.figure(figsize=[8,8])
plt.plot(np.array(y_train[-250:]))
plt.plot(y_one[-251:-1])
plt.xticks(xlabeltics[16:17], xlabels[16:17])
plt.title("One-step Ahead Prediction for Multiplicative Model (Last 250 samples)")
plt.xlabel('Time (hr)')
plt.ylabel('Load Demand')
plt.legend(['Train Set', "One-step Ahead"])
plt.grid()
plt.savefig('Multiplicative One Step Ahead (Last 250 samples)')
plt.show()


# Part b: Display the estimated variance of the error and the estimated covariance of the estimated parameters.
print(color.UNDERLINE + '\nDiagnostic Analysis_Part b:' + color.END)
print(f'\nThe variance of the error is:\n{var_e[0]}')

print(f'\nThe estimated covariance matrix of the estimated parameters is:\n{cov_theta_hat}')


# Part c: Is the derived model biased or this is an unbiased estimator?
print(color.UNDERLINE + '\nDiagnostic Analysis_Part c:' + color.END)
print(f'\nThe derived multiplicative model is biased because did not pass the whiteness test.')


############################################### Forecast Function ######################################################
# Part a: confidence intervals, zero/pole cancellation, chi-square test
print(color.BOLD + '\nForecast Function:' + color.END)

# Initialize the h-step forecast
y_hat_t_h = []

# Initialize the h-step forecast parts (MA and AR)
y_AR_h = []
y_MA_h = []

# Loop through the length of the test set
for h in range(1, len(y_test)):
    # Initialize the yt for each h
    yt = 0

    # Loop through the AR parameters
    for param_ar_index in range(1, len(AR_all.c)):
        # Check if the parameter was non-zero
        if AR_all.c[param_ar_index] != 0:
            # Check if the index of the parameter was less than h
            if param_ar_index < h:
                # Add to yt the multiplication of the parameter by previous y_hat
                yt += AR_all.c[param_ar_index] * y_hat_t_h[h - param_ar_index - 1]
            else:
                # Add to yt the multiplication of the parameter by the part of y_train
                yt += AR_all.c[param_ar_index] * np.array(y_train)[h - param_ar_index - 1]

    # Multiplu by -1 because will be taken to the other side and append
    y_AR_h.append(-1 * yt)

    # Initialize the et for each h
    et = 0

    # Check if h was more than the length of the MA parameters
    if h > len(MA_all.c):
        # Append zero
        y_MA_h.append(0)
    else:
        # loop from h to the last parameter of MA :
        for param_ma_index in range(h, len(MA_all.c)):
            # Check if the parameter was non-zero
            if MA_all.c[param_ma_index] != 0:
                # Update et
                et += MA_all.c[param_ma_index] * (np.array(y_train)[h - param_ma_index - 1] - y_one[h - param_ma_index - 2])

        # append et to MA_h
        y_MA_h.append(et)

    # Add y_AR_h and y_MA_h
    y_hat_t_h.append(y_AR_h[h - 1] + y_MA_h[h - 1])


############################################ h-step ahead Predictions###################################################
# Part a: confidence intervals, zero/pole cancellation, chi-square test
print(color.BOLD + '\nh-step ahead Predictions:' + color.END)

# Plotting the h-step ahead prediction
plt.figure(figsize=[20, 10])
plt.plot(test_index[1:], np.array(y_test)[1:])
plt.plot(test_index[1:], y_hat_t_h)
plt.xticks(xlabeltics[17:], xlabels[17:], fontsize= 12.5)
plt.title("h-step Forecast for Multiplicative Model",  fontsize= 20)
plt.xlabel('Time (hr)', fontsize= 15)
plt.ylabel('Load Demand', fontsize= 15)
plt.legend(['Test Set', "h-step"])
plt.grid()
plt.savefig('Multiplicative h-Step Ahead')
plt.show()

# Calculate the forecast error
error_cast = np.array(y_test)[1:] - y_hat_t_h

print(f"The variance of the forecast error is: \n{np.var(error_cast)}")

print(f"\nThe MSE for the multiplicative model is: \n{np.mean(np.square(error_cast))}")

