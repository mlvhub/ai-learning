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
# # Comparing Random Forest and XGBoost modeling performance

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# %%
# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ### Exercise 1: How many observations and features does the dataset have?
#

# %%
data

# %%
N_observations, N_features = X.shape
print('Number of Observations: ' + str(N_observations))
print('Number of Features: ' + str(N_features))


# %% [markdown]
# ### Initialize models
#
# In this step we define the number of base estimators, or individual trees, to be used in each model, and then intialize models for Random Forest regression and XGBoost regression. We'll just use the default parameters to make the performance comparisons. As a part of the performance comparison, we'll also measure the training times for both models.
#

# %%
# Initialize models
n_estimators=100
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)

# %%
# Fit models
# Measure training time for Random Forest
start_time_rf = time.time()
rf.fit(X_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf
rf_train_time


# %%
# Measure training time for XGBoost
start_time_xgb = time.time()
xgb.fit(X_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb
xgb_train_time

# %% [markdown]
# ### Exercise 2. Use the fitted models to make predictions on the test set. 
# Also, measure the time it takes for each model to make its predictions using the time.time() function to measure the times before and after each model prediction.
#

# %%
# Measure prediction time for Random Forest
start_time_rf = time.time()
y_pred_rf = rf.predict(X_test)
end_time_rf = time.time()
rf_pred_time = end_time_rf - start_time_rf
rf_pred_time


# %%
# Measure prediction time for XGBoost
start_time_xgb = time.time()
y_pred_xgb = xgb.predict(X_test)
end_time_xgb = time.time()
xgb_pred_time = end_time_xgb - start_time_xgb
xgb_pred_time

# %% [markdown]
# ### Exercise 3:  Calulate the MSE and R^2 values for both models
#

# %%
mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)


# %% [markdown]
# ### Exercise 4:  Print the MSE and R^2 values for both models
#

# %%
print("Random Forest Regression:")
print(f"MSE: {mse_rf:.4f}")
print(f"R^2: {r2_rf:.4f}")
print(f"Training Time: {rf_train_time:.4f} seconds")
print(f"Prediction Time: {rf_pred_time:.4f} seconds")

print("\nXGBoost Regression:")
print(f"MSE: {mse_xgb:.4f}")
print(f"R^2: {r2_xgb:.4f}")
print(f"Training Time: {xgb_train_time:.4f} seconds")
print(f"Prediction Time: {xgb_pred_time:.4f} seconds")

# %% [markdown]
# ### Exercise 5:  Print the timings for each model
#
# (Done above)

# %% [markdown]
# ### Exercise 6. Calculate the standard deviation of the test data
#

# %%
std_y = np.std(y_test)

# %% [markdown]
# ### Visualize the results
#

# %%
plt.figure(figsize=(14, 6))

# Random Forest plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="blue",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()


# XGBoost plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="orange",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(0,6)
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual Values")
plt.legend()
plt.tight_layout()

# %%
