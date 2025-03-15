# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multiple Linear Regression

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
file = "./FuelConsumptionCo2.csv"

df = pd.read_csv(file)
df.info()

# %% [markdown]
# ## Explore and select features

# %%
# Drop categoricals and any useless columns
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)

# %%
# We want to eliminate any strong dependencies or correlations between features by selecting the best one from each correlated group.
df.corr()

# %% [markdown]
# Look at the bottom row, which shows the correlation between each variable and the target, 'CO2EMISSIONS'. Each of these shows a fairly high level of correlation, each exceeding 85% in magnitude. Thus all of these features are good candidates. 
#
# Next, examine the correlations of the distinct pairs. 'ENGINESIZE' and 'CYLINDERS' are highly correlated, but 'ENGINESIZE' is more correlated with the target, so we can drop 'CYLINDERS'. 
#
# Similarly, each of the four fuel economy variables is highly correlated with each other. Since FUELCONSUMPTION_COMB_MPG is the most correlated with the target, we can drop the others: 'FUELCONSUMPTION_CITY,' 'FUELCONSUMPTION_HWY,' 'FUELCONSUMPTION_COMB.'
#
# > Notice that FUELCONSUMPTION_COMB and FUELCONSUMPTION_COMB_MPG are not perfectly correlated. They should be, though, because they measure the same property in different units. In practice, you would investigate why this is the case. You might find out that some or all of the data is not useable as is.

# %%
df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)
df.head(9)

# %% [markdown]
# To help with selecting predictive features that are not redundant, we consider the following scatter matrix, which shows the scatter plots for each pair of input features. The diagonal of the matrix shows each feature's histogram.
#

# %%
axes = pd.plotting.scatter_matrix(df, alpha=0.2)
# need to rotate axis labels so we can read them
for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)

# %% [markdown]
# As you can see, the relationship between 'FUELCONSUMPTION_COMB_MPG' and 'CO2EMISSIONS' is non-linear. In addition, you can clearly see three different curves. This suggests exploring the categorical variables to see if they are able to explain these differences.
#
# For now, let's just consider through modeling whether fuel economy explains some of the variances in the target as is.

# %% [markdown]
# ## Extract the input features and labels from the data set
#

# %%
X = df.iloc[:,[0,1]].to_numpy()
y = df.iloc[:,[2]].to_numpy()

# %% [markdown]
# ## Preprocess selected features
#
# We should standardise our input features so the model doesn't inadvertently favor any feature due to its magnitude.
# The typical way to do this is to subtract the mean and divide by the standard deviation.
#
#  Scikit-learn can do this for us.
#

# %%
from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()
X_std = std_scaler.fit_transform(X)

# %% [markdown]
# > In practice, if you want to properly evaluate your model, you should definitely not apply such operations to the entire dataset but to the train and test data separately.

# %%
pd.DataFrame(X_std).describe().round(2)

# %% [markdown]
# As we can see, a standardised variable has zero mean and a standard deviation of one.
#

# %% [markdown]
# ## Create train and test datasets
#

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.2,random_state=42)

# %% [markdown]
# ## Build a multiple linear regression model
#

# %%
from sklearn import linear_model

# create a model object
regressor = linear_model.LinearRegression()

# train the model in the training data
regressor.fit(X_train, y_train)

# Print the coefficients
coef_ =  regressor.coef_
intercept_ = regressor.intercept_

print ('Coefficients: ',coef_)
print ('Intercept: ',intercept_)


# %% [markdown]
# The Coefficients and Intercept parameters define the best-fit hyperplane to the data. Since there are only two variables, hence two parameters, the hyperplane is a plane. But this best-fit plane will look different in the original, unstandardised feature space. 

# %% [markdown]
# We can transform our model's parameters back to the original space prior to standardization as follows. This gives us a proper sense of what they mean in terms of our original input features. Without these adjustments, the model's outputs would be tied to an abstract, transformed space that doesn’t align with the actual independent variables and the real-world problem we’re solving.

# %%
# Get the standard scaler's mean and standard deviation parameters
means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

# The least squares parameters can be calculated relative to the original, unstandardized feature space as:
coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

print ('Coefficients: ', coef_original)
print ('Intercept: ', intercept_original)


# %% [markdown]
# One would expect that for the limiting case of zero ENGINESIZE and zero FUELCONSUMPTION_COMB_MPG, the resulting CO2 emissions should also be zero. This is inconsistent with the 'best fit' hyperplane, which has a non-zero intercept of 329 g/km. The answer must be that the target variable does not have a very strong linear relationship to the dependent variables, and/or the data has outliers that are biasing the result. Outliers can be handled in preprocessing, or as we will learn about later in the course, by using regularization techniques. One or more of the variables might have a nonlinear relationship to the target. Or there may still be some colinearity amongst the input variables.
#

# %% [markdown]
# ## Visualize model outputs
#

# %%
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Ensure X1, X2, and y_test have compatible shapes for 3D plotting
X1 = X_test[:, 0] if X_test.ndim > 1 else X_test
X2 = X_test[:, 1] if X_test.ndim > 1 else np.zeros_like(X1)

# Create a mesh grid for plotting the regression plane
x1_surf, x2_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), 
                               np.linspace(X2.min(), X2.max(), 100))

y_surf = intercept_ +  coef_[0,0] * x1_surf  +  coef_[0,1] * x2_surf

# Predict y values using trained regression model to compare with actual y_test for above/below plane colors
y_pred = regressor.predict(X_test.reshape(-1, 1)) if X_test.ndim == 1 else regressor.predict(X_test)
above_plane = y_test >= y_pred
below_plane = y_test < y_pred
above_plane = above_plane[:,0]
below_plane = below_plane[:,0]

# Plotting
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points above and below the plane in different colors
ax.scatter(X1[above_plane], X2[above_plane], y_test[above_plane],  label="Above Plane",s=70,alpha=.7,ec='k')
ax.scatter(X1[below_plane], X2[below_plane], y_test[below_plane],  label="Below Plane",s=50,alpha=.3,ec='k')

# Plot the regression plane
ax.plot_surface(x1_surf, x2_surf, y_surf, color='k', alpha=0.21,label='plane')

# Set view and labels
ax.view_init(elev=10)

ax.legend(fontsize='x-large',loc='upper center')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(None, zoom=0.75)
ax.set_xlabel('ENGINESIZE', fontsize='xx-large')
ax.set_ylabel('FUELCONSUMPTION', fontsize='xx-large')
ax.set_zlabel('CO2 Emissions', fontsize='xx-large')
ax.set_title('Multiple Linear Regression of CO2 Emissions', fontsize='xx-large')
plt.tight_layout()

# %% [markdown]
# Instead of making a 3D plot, which is difficult to interpret, we can look at vertical slices of the 3D plot by plotting each variable separately as a best-fit line using the corresponding regression parameters.
#

# %%
plt.scatter(X_train[:,0], y_train,  color='blue')
plt.plot(X_train[:,0], coef_[0,0] * X_train[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# %%
plt.scatter(X_train[:,1], y_train,  color='blue')
plt.plot(X_train[:,1], coef_[0,1] * X_train[:,1] + intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")

# %% [markdown]
# Evidently, the solution is incredibly poor because the model is trying to fit a plane to a non-planar surface.
#

# %% [markdown]
# ## Practice exercises
#

# %% [markdown]
# ### 1. Determine and print the parameters for the best-fit linear regression line for CO2 emission with respect to engine size

# %%
# only use ENGINESIZE
X_train_1 = X_train[:,[0]]

# create a model object
regressor = linear_model.LinearRegression()

# train the model in the training data
regressor.fit(X_train_1, y_train)

# Print the coefficients
coef_ =  regressor.coef_
intercept_ = regressor.intercept_

print ('Coefficients: ',coef_)
print ('Intercept: ',intercept_)

# %% [markdown]
# ### 2. Produce a scatterplot of CO2 emission against ENGINESIZE and include the best-fit regression line to the training data

# %%
plt.scatter(X_train_1[:,0], y_train,  color='blue')
plt.plot(X_train_1[:,0], coef_[0,0] * X_train_1[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# %% [markdown]
# ### 3. Generate the same scatterplot and best-fit regression line, but now base the result on the test data set

# %%
plt.scatter(X_test[:,0], y_test,  color='blue')
plt.plot(X_test[:,0], coef_[0,0] * X_test[:,0] + intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# %% [markdown]
# 4. Repeat the same modeling but use FUELCONSUMPTION_COMB_MPG as the independent variable instead. Display the model coefficients including the intercept

# %%
X_train_2 = X_train[:,[1]]

# create a model object
regressor = linear_model.LinearRegression()

# train the model in the training data
regressor.fit(X_train_2, y_train)

# Print the coefficients
coef_ =  regressor.coef_
intercept_ = regressor.intercept_

print ('Coefficients: ',coef_)
print ('Intercept: ',intercept_)

# %% [markdown]
# ### 5. Generate a scatter plot showing the results as before on the test data.

# %%
plt.scatter(X_train_2[:,0], y_train,  color='blue')
plt.plot(X_train_2[:,0], coef_[0,0] * X_train_2[:,0] + intercept_[0], '-r')
plt.xlabel("FUELCONSUMPTION_COMB_MPG")
plt.ylabel("Emission")
