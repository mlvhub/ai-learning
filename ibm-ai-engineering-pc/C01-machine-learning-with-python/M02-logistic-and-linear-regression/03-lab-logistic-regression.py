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
# # Logistic Regression with Python

# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


# %% [markdown]
# ## Classification with Logistic Regression
#

# %%
file = "./ChurnData.csv"

churn_df = pd.read_csv(file)
churn_df.info()

# %%
churn_df.head()

# %% [markdown]
# ## Data Preprocessing
#

# %%
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df

# %% [markdown]
# For modeling the input fields X and the target field y need to be fixed. Since that the target to be predicted is 'churn', the data under this field will be stored under the variable 'y'. We may use any combination or all of the remaining fields as the input.

# %%
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]  #print the first 5 values

# %%
y = np.asarray(churn_df['churn'])
y[0:5] #print the first 5 values

# %% [markdown]
# It is also a norm to standardize or normalize the dataset in order to have all the features at the same scale. This helps the model learn faster and improves the model performance. We may make use of StandardScalar function in the Scikit-Learn library.
#

# %%
X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]

# %% [markdown]
# The trained model has to be tested and evaluated on data which has not been used during training. Therefore, it is required to separate a part of the data for testing and the remaining for training.

# %%
X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ## Logistic Regression Classifier modeling
#
# Let's build the model using __LogisticRegression__ from the Scikit-learn package and fit our model with train data set.
#

# %%
LR = LogisticRegression().fit(X_train,y_train)
LR

# %% [markdown]
# Fitting, or in simple terms training, gives us a model that has now learnt from the traning data and can be used to predict the output variable. Let us predict the churn parameter for the test data set.

# %%
yhat = LR.predict(X_test)
yhat[:10]

# %% [markdown]
# To understand this prediction, we can also have a look at the prediction probability of data point of the test data set. 

# %%
yhat_prob = LR.predict_proba(X_test)
yhat_prob[:10]

# %% [markdown]
# The first column is the probability of the record belonging to class 0, and second column that of class 1.
#
# > Note that the class prediction system uses the threshold for class prediction as 0.5. This means that the class predicted is the one which is most likely.

# %% [markdown]
# Since the purpose here is to predict the 1 class more acccurately, we can also examine what role each input feature has to play in the prediction of the 1 class.

# %%
coefficients = pd.Series(LR.coef_[0], index=churn_df.columns[:-1])
coefficients.sort_values().plot(kind='barh')
plt.title("Feature Coefficients in Logistic Regression Churn Model")
plt.xlabel("Coefficient Value")

# %% [markdown]
# Large positive value of LR Coefficient for a given field indicates that increase in this parameter will lead to better chance of a positive, i.e. 1 class. A large negative value indicates the opposite, which means that an increase in this parameter will lead to poorer chance of a positive class. A lower absolute value indicates weaker effect of the change in that field on the predicted class. 

# %% [markdown]
# ## Performance Evaluation
#
# Once the predictions have been generated, it becomes prudent to evaluate the performance of the model in predicting the target variable.

# %% [markdown]
# ### log loss
#
# Log loss (Logarithmic loss), also known as Binary Cross entropy loss, is a function that generates a loss value based on the class wise prediction probabilities and the actual class labels. The lower the log loss value, the better the model is considered to be.
#

# %%
log_loss(y_test, yhat_prob)

# %% [markdown]
# ## Practice Exercises
#

# %% [markdown]
# 1. Let us assume we add the feature 'callcard' to the original set of input features. What will the value of log loss be in this case? 

# %%
churn_df = pd.read_csv(file)
churn_df = churn_df[['callcard', 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['callcard', 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

X_norm = StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

log_loss(y_test, yhat_prob)

# %% [markdown]
# 2. Let us assume we add the feature 'wireless' to the original set of input features. What will the value of log loss be in this case? 

# %%
churn_df = pd.read_csv(file)
churn_df = churn_df[['wireless', 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['wireless', 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

X_norm = StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

log_loss(y_test, yhat_prob)

# %% [markdown]
# 3. What happens to the log loss value if we add both "callcard" and "wireless" to the input features? 

# %%
churn_df = pd.read_csv(file)
churn_df = churn_df[['callcard', 'wireless', 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['callcard', 'wireless', 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

X_norm = StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

log_loss(y_test, yhat_prob)

# %% [markdown]
# 4. What happens to the log loss if we remove the feature 'equip' from the original set of input features? 

# %%
churn_df = pd.read_csv(file)
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ']])
y = np.asarray(churn_df['churn'])

X_norm = StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

log_loss(y_test, yhat_prob)

# %% [markdown]
# 5. What happens to the log loss if we remove the features 'income' and 'employ' from the original set of input features? 

# %%
churn_df = pd.read_csv(file)
churn_df = churn_df[['tenure', 'age', 'address', 'ed', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

X = np.asarray(churn_df[['tenure', 'age', 'address', 'ed', 'equip']])
y = np.asarray(churn_df['churn'])

X_norm = StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split( X_norm, y, test_size=0.2, random_state=4)

LR = LogisticRegression().fit(X_train,y_train)
yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)

log_loss(y_test, yhat_prob)
