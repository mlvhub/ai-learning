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
# # Simple Linear Regression

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Import data and explore

# %%
file = "./FuelConsumptionCo2.csv"

df = pd.read_csv(file)
df.info()

# %%
df.sample(5)

# %% [markdown]
# Data source: https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
#
# Schema:
# - **MODEL YEAR** e.g. 2014
# - **MAKE** e.g. VOLVO
# - **MODEL** e.g. S60 AWD
# - **VEHICLE CLASS** e.g. COMPACT
# - **ENGINE SIZE** e.g. 3.0
# - **CYLINDERS** e.g 6
# - **TRANSMISSION** e.g. AS6
# - **FUEL TYPE** e.g. Z
# - **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 13.2
# - **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 9.5
# - **FUEL CONSUMPTION COMBINED (L/100 km)** e.g. 11.5
# - **FUEL CONSUMPTION COMBINED MPG (MPG)** e.g. 25
# - **CO2 EMISSIONS (g/km)** e.g. 182 

# %%
df.describe()

# %%
# Select a few features that might be indicative of CO2 emission to explore more.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.sample(9)

# %%
# Visualise the features
viz = cdf[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()

# %% [markdown]
# Observations:
# - most engines are 4, 6 or 8 cylinders
# - most engine sizes 2 and 4 litres
# - combined fuel comsumption and CO2 emissions have similar distributions

# %%
# Plot the relationship between fuel consumption and CO2 emissions
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")

# %% [markdown]
# Three car groups each have a strong linear relationship between their combined fuel consumption and their CO2 emissions. 
# Their intercepts are similar, while they noticeably differ in their slopes.
#

# %%
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.xlim(0,27)

# %% [markdown]
# Although the relationship between engine size and CO2 emission is quite linear, you can see that their correlation is weaker than that for each of the three fuel consumption groups. 
#
# > Notice that the x-axis range has been expanded to make the two plots more comparable.

# %%
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")

# %% [markdown]
# ## Extract the input feature and labels from the dataset
#
# For illustration purposes, we will use engine size to predict CO2 emission with a linear regression model.

# %%
X = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# %% [markdown]
# ### Create train and test datasets
#
# We will randomly split the dataset into training and test sets, using 80% of the data for training and the remaining 20% for testing.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
type(X_train), np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)

# %% [markdown]
# ### Build a simple linear regression model

# %%
from sklearn import linear_model

# create a model object
regressor = linear_model.LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print the coefficients
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)

# %% [markdown]
# Here, __Coefficient__ and __Intercept__ are the regression parameters determined by the model.  
#
# They define the slope and intercept of the 'best-fit' line to the training data. 
#

# %% [markdown]
# ### Visualize model outputs
#
# We can visualise the goodness-of-fit of the model to the training data by plotting the fitted line over the data.
#
# The regression model is the line given by y = intercept + coefficient * x.

# %%
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# %% [markdown]
# ### Model evaluation
#
# We can compare the actual values and predicted values of the model to evaluate its accuracy.
#
# Evaluation metrics play a key role in the development of machine learning models. They provide insight into areas that require improvement.
#
# For regression problems, we can use the following metrics:
# - Mean Absolute Error (MAE): it is the mean of the absolute value of the errors.
# - Mean Squared Error (MSE): mean of the squared errors.
# - Root Mean Squared Error (RMSE): square root of MSE, simply transforms MSE into the same unit as the dependent variable.
# - R-squared (R2): not an error, but a goodness-of-fit metric. It represents how close the data points are to the fitted regression line.

# %% [markdown]
#

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))

print("Mean absolute error: %.2f" % mean_absolute_error(y_test_, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))
print("Root mean squared error: %.2f" % root_mean_squared_error(y_test_, y_test))
print("R2-score: %.2f" % r2_score( y_test_, y_test) )

# %% [markdown]
# ## Practice exercises

# %% [markdown]
# ### 1. Plot the regression model result over the test data instead of the training data. Visually evaluate whether the result is good.
#

# %%
plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, regressor.coef_ * X_test + regressor.intercept_, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# %% [markdown]
# ### 2. Select the fuel consumption feature from the dataframe and split the data 80%/20% into training and testing sets. 
#

# %%
X = cdf.FUELCONSUMPTION_COMB.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
type(X_train), np.shape(X_train), np.shape(X_test), np.shape(y_train), np.shape(y_test)

# %% [markdown]
# ### 3.  Train a linear regression model using the training data you created.
#

# %%
from sklearn import linear_model

# create a model object
regressor = linear_model.LinearRegression()

# train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(X_train.reshape(-1, 1), y_train)

# Print the coefficients
print ('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print ('Intercept: ',regressor.intercept_)

# %%
plt.scatter(X_train, y_train,  color='blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, '-r')
plt.xlabel("Fuel Comsumption")
plt.ylabel("Emission")

# %% [markdown]
# ### 4. Use the model to make test predictions on the fuel consumption testing data.
#

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Use the predict method to make test predictions
y_test_ = regressor.predict(X_test.reshape(-1,1))
y_test_

# %% [markdown]
# ### 5. Calculate and print the Mean Squared Error of the test predictions.
#

# %%
print("Mean absolute error: %.2f" % mean_absolute_error(y_test_, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test_, y_test))
print("Root mean squared error: %.2f" % root_mean_squared_error(y_test_, y_test))
print("R2-score: %.2f" % r2_score( y_test_, y_test) )
