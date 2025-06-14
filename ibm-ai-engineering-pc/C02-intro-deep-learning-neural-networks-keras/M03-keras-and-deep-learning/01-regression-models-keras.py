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
# # Regression Models with Keras

# %%
# suppress the warning messages due to use of CPU architechture for TensorFlow.
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
import pandas as pd
import numpy as np
import keras

import warnings
warnings.simplefilter('ignore', FutureWarning)

# %% [markdown]
# ## Download and Clean the Data Set
#

# %%
filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

concrete_data.head()

# %%
concrete_data.shape

# %% [markdown]
# > Because of the few samples (1000~), we have to be careful not to overfit the training data.

# %%
concrete_data.describe()

# %%
# check for nulls
concrete_data.isnull().sum()

# %%
# Split data into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
predictors

# %%
predictors.head()

# %%
target = concrete_data['Strength'] # Strength column
target

# %%
target.head()

# %%
# normalize the data by substracting the mean and dividing by the standard deviation
predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

# %%
n_cols = predictors_norm.shape[1] # number of predictors
n_cols

# %% [markdown]
# ##  Import Keras Packages
#

# %%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input


# %% [markdown]
# ## Build a Neural Network
#

# %%
# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# %% [markdown]
# ## Train and Test the Network
#

# %%
# build the model
model = regression_model()

# %%
# fit the model, leave out 30% of the data for validation and we will train the model for 100 epochs
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

# %% [markdown]
# ### Practice Exercise 1

# %%
model2 = Sequential()
model2.add(Input(shape=(n_cols,)))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(50, activation='relu'))
model2.add(Dense(1))


# %% [markdown]
# ### Practice Exercise 2

# %%
# compile model
model2.compile(optimizer='adam', loss='mean_squared_error')

# fit the model
model2.fit(predictors_norm, target, validation_split=0.1, epochs=100, verbose=2)

# %%
