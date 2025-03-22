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
# # K-Nearest Neighbors Classifier

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## About the data set

# %% [markdown]
# Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset,  with predefined labels, we need to build a model to be used to predict class of a new or unknown case.
#
# The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.
#
# The target field, called **custcat**, has four possible service categories that correspond to the four customer groups, as follows:
#
# 1. Basic Service
# 2. E-Service
# 3. Plus Service
# 4. Total Service
#
# Our objective is to build a classifier to predict the service category for unknown cases. We will use a specific type of classification called K-nearest neighbors.
#

# %%
path = './teleCust1000t.csv'

# %%
df = pd.read_csv(path)
df.head()

# %% [markdown]
# ## Data Visualization and Analysis

# %%
df['custcat'].value_counts()

# %%
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# %%
correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
correlation_values

# %%
# separate the data into the input data set and the target data set
X = df.drop('custcat',axis=1)
y = df['custcat']

# %% [markdown]
# ## Normalize Data

# %% [markdown]
# KNN makes predictions based on the distance between data points (samples), i.e. for a given test point, the algorithm finds the k-nearest neighbors by measuring the distance between the test point and other data points in the dataset.
#
# By normalizing / standardizing the data, we ensure that all features contribute equally to the distance calculation. Since normalization scales each feature to have zero mean and unit variance, it puts all features on the same scale (with no feature dominating due to its larger range).
#
# This helps KNN make better decisions based on the actual relationships between features, not just on the magnitude of their values.
#

# %%
X_norm = StandardScaler().fit_transform(X)

# %% [markdown]
# ### Train Test Split
#

# %% [markdown]
# We can retain 20% of the data for testing purposes and use the rest for training. Assigning a random state ensures reproducibility of the results across multiple executions.

# %%
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

# %% [markdown]
# ## KNN Classification
#

# %%
k = 3
#Train Model and Predict  
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

# %%
yhat = knn_model.predict(X_test)

# %%
print("Test set Accuracy: ", accuracy_score(y_test, yhat))

# %% [markdown]
# ### Exercise 1
# Can you build the model again, but this time with k=6?
#

# %%
k = 6
#Train Model and Predict  
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

yhat = knn_model.predict(X_test)

print("Test set Accuracy: ", accuracy_score(y_test, yhat))

# %% [markdown]
# ### Choosing the correct value of k
#

# %% [markdown]
# K in KNN, is the number of nearest neighbors to examine. However, the choice of the value of 'k' clearly affects the model. Therefore, the appropriate choice of the value of the variable `k` becomes an important task. 
#
# The general way of doing this is to train the model on a set of different values of k and noting the performance of the trained model on the testing set. The model with the best value of `accuracy_score` is the one with the ideal value of the parameter k.

# %%
Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

# %%
plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()

# %%
print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 

# %% [markdown]
# However, since this graph is still rising, there can be a chance that the model will give a better performance with an even higher value of k.
#

# %% [markdown]
#
# ### Exercise 2
# Run the training model for 30 values of k and then again for 100 values of k. Identify the value of k that best suits this data and the accuracy on the test set for this model.

# %%
Ks = 30
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 

# %%
Ks = 100
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1) 

# %%
plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()

# %% [markdown]
# ### Exercise 4
#
# Can you justify why the model performance on training data is deteriorating with increase in the value of k?
#

# %% [markdown]
# At some point you start including more and more unrelated data.

# %% [markdown]
# ### Exercise 5
# We can see that even the with the optimum values, the KNN model is not performing that well on the given data set. Can you think of the possible reasons for this?
#

# %% [markdown]
# - not much correlation between features and target
# - not enough data
# - no clear separation between classes

# %%
