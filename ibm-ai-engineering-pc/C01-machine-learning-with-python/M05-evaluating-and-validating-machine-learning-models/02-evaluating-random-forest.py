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
# # Evaluating Random Forest Performance

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew

# %%
# Load the dataset
data = fetch_california_housing()
X, y = data.data, data.target
X, y

# %%
print(data.DESCR)

# %% [markdown]
# ### Exercise 1. Split the data into training and testing sets
#

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## Explore the training data
#

# %%
eda = pd.DataFrame(data=X_train)
eda.columns = data.feature_names
eda['MedHouseVal'] = y_train
eda.describe()

# %% [markdown]
# ### Exercise 2. What range are most of the median house prices valued at?
#

# %% [markdown]
# $119,000 - $265,125

# %% [markdown]
# ### How are the median house prices distributed?
#

# %%
# Plot the distribution
plt.hist(1e5*y_train, bins=30, color='lightblue', edgecolor='black')
plt.title(f'Median House Value Distribution\nSkewness: {skew(y_train):.2f}')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')

# %% [markdown]
# Evidently the distribution is skewed and there are quite a few clipped values at around $500,000. 
#

# %% [markdown]
# ### Model fitting and prediction
#

# %%
# Initialize and fit the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_test = rf_regressor.predict(X_test)

# %% [markdown]
# ### Estimate out-of-sample MAE, MSE, RMSE, and R²
#

# %%
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# %% [markdown]
# ### Exercise 3. What do these statistics mean to you?
# How comfortable could you be with stopping here and communicating the results to the C-suite?
#

# %% [markdown]
# No.
# The mean absolute error is $32,760, meaning on average the predirected median house prices are off by $32,760.
# An R² of 0.805 is not good enough, it means the model explains 80.5% of the variance, which isn't great.

# %% [markdown]
# ### Plot Actual vs Predicted values
#

# %%
plt.scatter(y_test, y_pred_test, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest Regression - Actual vs Predicted")
plt.show()

# %% [markdown]
# ### Exercise 4. Plot the histogram of the residual errors (dollars)
#

# %%
# Enter your code here
residuals = (y_test - y_pred_test)
plt.hist(residuals, bins=30, color='lightblue', edgecolor='black')
plt.title('Histogram of Residual Errors')
plt.xlabel('Residual Error (dollars)')
plt.ylabel('Frequency')
plt.show()
### Exercise 5. What do you think about the model's performance?



# %% [markdown]
# ### Exercise 5. Plot the model residual errors by median house value.
#

# %%
residuals_df = pd.DataFrame({
    'Actual': 1e5*y_test,
    'Residuals': residuals
})

# Sort the DataFrame by the actual target values
residuals_df = residuals_df.sort_values(by='Actual')

# Plot the residuals
plt.scatter(residuals_df['Actual'], residuals_df['Residuals'], marker='o', alpha=0.4,ec='k')
plt.title('Median House Value Prediciton Residuals Ordered by Actual Median Prices')
plt.xlabel('Actual Values (Sorted)')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# %% [markdown]
# ### Exercise 6. What trend can you infer from this residual plot?
#

# %% [markdown]
# ### Exercise 7. Display the feature importances as a bar chart.
#

# %%
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names

# Plot feature importances
plt.bar(range(X.shape[1]), importances[indices],  align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Regression")
plt.show()

# %% [markdown]
# ### Exercise 8. Some final thoughts to consider
#
# - Will the skewness affect the performance of Random Forest regression?
# - Does the clipping of median house prices above \$500,000 bias the predictions?
# - Also, do we need to standardize the data?
#
