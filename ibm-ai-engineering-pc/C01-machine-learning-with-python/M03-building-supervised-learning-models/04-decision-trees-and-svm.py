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
# # Credit Card Fraud Detection with Decision Trees and SVM
#

# %%
# Import the libraries we need to use in this lab
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore')

# %%
# CSV too large to upload to GitHub
#url = './creditcard.csv'
url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"


# %%
# read the input data
raw_data=pd.read_csv(url).dropna()
raw_data

# %% [markdown]
# ## Dataset Analysis

# %% [markdown]
# Each row in the dataset represents a credit card transaction. As shown above, each row has 31 variables. One variable (the last variable in the table above) is called Class and represents the target variable.
#
# Note: For confidentiality reasons, the original names of most features are anonymized V1, V2 .. V28. The values of these features are the result of a PCA transformation and are numerical.
#
# Source: https://www.kaggle.com/mlg-ulb/creditcardfraud

# %%
# get the set of distinct classes
labels = raw_data.Class.unique()
labels

# %%

# get the count of each class
sizes = raw_data.Class.value_counts().values
sizes

# %%
# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')

# %%
correlation_values = raw_data.corr()['Class'].drop('Class')
correlation_values.plot(kind='barh', figsize=(10, 6))

# %% [markdown]
# ## Dataset Preprocessing

# %% [markdown]
# We will apply standard scaling to the input features and normalize them using $L_1$ norm for the training models to converge quickly.

# %%
# standardize features by removing the mean and scaling to unit variance
raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(raw_data.iloc[:, 1:30])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

# %% [markdown]
# ## Dataset Train/Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# ## Build a Decision Tree Classifier model with Scikit-Learn

# %%
w_train = compute_sample_weight('balanced', y_train)

# %%
# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)

dt.fit(X_train, y_train, sample_weight=w_train)

# %% [markdown]
# ## Build a Support Vector Machine model with Scikit-Learn

# %% [markdown]
# Unlike Decision Trees, we do not need to initiate a separate sample_weight for SVMs. We can simply pass a parameter in the scikit-learn function.
#

# %%
# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate the Decision Tree Classifier Models

# %%
# probabilities of the test samples belonging to the class of fraudulent transactions
y_pred_dt = dt.predict_proba(X_test)[:,1]

# %% [markdown]
# Using these probabilities, we can evaluate the Area Under the Receiver Operating Characteristic Curve (ROC-AUC) score as a metric of model performance. 
# The AUC-ROC score evaluates your model's ability to distinguish positive and negative classes considering all possible probability thresholds. The higher its value, the better the model is considered for separating the two classes of values.
#

# %%
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

# %% [markdown]
# ## Evaluate the Support Vector Machine Models

# %%
# compute the probabilities of the test samples belonging to the class of fraudulent transactions
y_pred_svm = svm.decision_function(X_test)

# %%
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

# %% [markdown]
# ## Practice Exercises
#

# %% [markdown]
# Q1. Currently, we have used all 30 features of the dataset for training the models. Use the `corr()` function to find the top 6 features of the dataset to train the models on. 
#

# %%
correlation_values = raw_data.corr()['Class'].drop('Class').sort_values(ascending=False)[:6].index.tolist()
correlation_values


# %% [markdown]
# Q2. Using only these 6 features, modify the input variable for training.
#

# %%
# standardize features by removing the mean and scaling to unit variance
raw_data[correlation_values] = StandardScaler().fit_transform(raw_data[correlation_values])
data_matrix = raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:,[3,10,12,14,16,17]]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# Q3. Execute the Decision Tree model for this modified input variable. How does the value of ROC-AUC metric change?
#

# %%
w_train = compute_sample_weight('balanced', y_train)

# for reproducible output across multiple function calls, set random_state to a given integer value
dt = DecisionTreeClassifier(max_depth=4, random_state=35)

dt.fit(X_train, y_train, sample_weight=w_train)

# probabilities of the test samples belonging to the class of fraudulent transactions
y_pred_dt = dt.predict_proba(X_test)[:,1]

roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))


# %% [markdown]
# Q4. Execute the SVM model for this modified input variable. How does the value of ROC-AUC metric change?
#

# %%
# for reproducible output across multiple function calls, set random_state to a given integer value
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)

# compute the probabilities of the test samples belonging to the class of fraudulent transactions
y_pred_svm = svm.decision_function(X_test)

roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

# %% [markdown]
# Q5. What are the inferences you can draw about Decision Trees and SVMs with what you have learnt in this lab?
#

# %% [markdown]
# - With a larger set of features, SVM performed relatively better in comparison to the Decision Trees.
# - Decision Trees benefited from feature selection and performed better.
# - SVMs may require higher feature dimensionality to create an efficient decision hyperplane.

# %%
