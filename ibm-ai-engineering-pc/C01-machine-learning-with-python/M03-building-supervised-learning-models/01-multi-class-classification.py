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
# # Multi-Class Classification

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Dataset Overview
#
# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg .tg-7zrl{text-align:left;vertical-align:bottom}
# </style>
# <table class="tg"><thead>
#   <tr>
#     <th class="tg-7zrl">Variable Name</th>
#     <th class="tg-7zrl">Type</th>
#     <th class="tg-7zrl">Description</th>
#   </tr></thead>
# <tbody>
#   <tr>
#     <td class="tg-7zrl">Gender</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Age</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Height</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Weight</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">family_history_with_overweight</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Has a family member suffered or suffers from overweight?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FAVC</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you eat high caloric food frequently?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FCVC</td>
#     <td class="tg-7zrl">Integer</td>
#     <td class="tg-7zrl">Do you usually eat vegetables in your meals?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">NCP</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How many main meals do you have daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CAEC</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Do you eat any food between meals?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">SMOKE</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you smoke?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CH2O</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How much water do you drink daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">SCC</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you monitor the calories you eat daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FAF</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How often do you have physical activity?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">TUE</td>
#     <td class="tg-7zrl">Integer</td>
#     <td class="tg-7zrl">How much time do you use technological devices such as cell phone, videogames, television, computer and others?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CALC</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">How often do you drink alcohol?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">MTRANS</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Which transportation do you usually use?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">NObeyesdad</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Obesity level</td>
#   </tr>
# </tbody></table>
#

# %%
file_path = './Obesity-level-prediction-dataset.csv'
data = pd.read_csv(file_path)
data.head()

# %% [markdown]
# ## EDA

# %%
# Distribution of target variable
sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')

# %% [markdown]
# The dataset seems fairly balanced and does not require any special attention in terms of biased training.

# %% [markdown]
# ## Exercise 1

# %%
data.isnull().sum()

# %%
data.info()

# %%
data.describe()

# %% [markdown]
# ## Preprocessing the data
#

# %% [markdown]
# ### Feature scaling
# Scale the numerical features to standardize their ranges for better model performance.
#

# %%
# Standardizing continuous numerical features
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
continuous_columns

# %%
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])
scaled_features

# %%
# Converting to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
scaled_df

# %%
# Combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)
scaled_data

# %% [markdown]
# Standardization of data is important to better define the decision boundaries between classes by making sure that the feature variations are in similar scales.

# %% [markdown]
# ### One-hot encoding
# Convert categorical variables into numerical format using one-hot encoding.
#

# %%
# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column
categorical_columns

# %%
# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
encoded_features

# %%
# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
encoded_df


# %%
# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
prepped_data

# %% [markdown]
# We can see that all the categorical variables have now been modified to one-hot encoded features. This increases the overall number of fields to 24. 
#

# %% [markdown]
# ### Encode the target variable
#

# %%
# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()

# %% [markdown]
# ### Separate the input and target data
#

# %%
# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']

# %% [markdown]
# ## Model training and evaluation 
#

# %% [markdown]
# ### Splitting the data set
# Split the data into training and testing subsets.
#

# %%
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% [markdown]
# ### Logistic Regression with One-vs-All

# %% [markdown]
# In the One-vs-All approach:
#
# * The algorithm trains a single binary classifier for each class.
# * Each classifier learns to distinguish a single class from all the others combined.
# * If there are k classes, k classifiers are trained.
# * During prediction, the algorithm evaluates all classifiers on each input, and selects the class with the highest confidence score as the predicted class.
#
# #### Advantages:
# * Simpler and more efficient in terms of the number of classifiers (k)
# * Easier to implement for algorithms that naturally provide confidence scores (e.g., logistic regression, SVM).
#
# #### Disadvantages:
# * Classifiers may struggle with class imbalance since each binary classifier must distinguish between one class and the rest.
# * Requires the classifier to perform well even with highly imbalanced datasets, as the "all" group typically contains more samples than the "one" class.

# %% [markdown]
# Train a logistic regression model using the One-vs-All strategy and evaluate its performance.
#

# %%
# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)

# %% [markdown]
# Evaluate the accuracy of the trained model as a measure of its performance on unseen testing data.

# %%
# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

# %% [markdown]
# ### Logistic Regression with One-vs-One

# %% [markdown]
#
# In the One-vs-One approach:
# * The algorithm trains a binary classifier for every pair of classes in the dataset.
# * If there are k classes, this results in $k(k-1)/2$ classifiers.
# * Each classifier is trained to distinguish between two specific classes, ignoring the rest.
# * During prediction, all classifiers are used, and a "voting" mechanism decides the final class by selecting the class that wins the majority of pairwise comparisons.
#
# #### Advantages:
# * Suitable for algorithms that are computationally expensive to train on many samples because each binary classifier deals with a smaller dataset (only samples from two classes).
# * Can be more accurate in some cases since classifiers focus on distinguishing between two specific classes at a time.
#
# #### Disadvantages:
# * Computationally expensive for datasets with a large number of classes due to the large number of classifiers required.
# * May lead to ambiguous predictions if voting results in a tie.
#

# %%
# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)

# %%
# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

# %% [markdown]
# ## Exercises

# %% [markdown]
# Q1. Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3) and observe the impact on model performance.
#

# %%
sizes = [0.1, 0.3, 0.5, 0.7, 0.9]

data_per_size=[]

for size in sizes:
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42, stratify=y)
    data_per_size.append((size, (X_train, X_test, y_train, y_test)))

data_per_size

# %%
Y_preds_per_size=[]

for size, data in data_per_size:
    X_train, X_test, y_train, y_test = data

    # Training logistic regression model using One-vs-All (default)
    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)

    # Training logistic regression model using One-vs-One
    model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
    model_ovo.fit(X_train, y_train)

    # Predictions
    y_pred_ova = model_ova.predict(X_test)
    y_pred_ovo = model_ovo.predict(X_test)
    Y_preds_per_size.append((size, y_test, y_pred_ova, y_pred_ovo))


# %%
for size, y_test, y_pred_ova, y_pred_ovo in Y_preds_per_size:
    print(f"Test size: {size} Ova: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}% Ovo: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

# %% [markdown]
# Q2. Plot a bar chart of feature importance using the coefficients from the One vs All logistic regression model. Also try for the One vs One model.
#

# %%
feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("OvA Feature Importance")
plt.xlabel("Importance")


# %%
# for i, classifier in enumerate(ovo_classifier.estimators_):
#     print(f"Coefficients for classifier {i}:")
#     print(classifier.coef_)

# feature_importance = np.mean(np.abs(model_ovo.coef_), axis=0)
# plt.barh(X.columns, feature_importance)
# plt.title("OvO Feature Importance")
# plt.xlabel("Importance")


# %% [markdown]
# Q3. Write a function `obesity_risk_pipeline` to automate the entire pipeline: <br>
# <ol>
# <li> Loading and preprocessing the data </li>
# <li> Training the model </li>
# <li> Evaluating the model </li>
# </ol>
# The function should accept the file path and test set size as the input arguments.
#

# %%
def obesity_risk_pipeline(data_path, test_size):
    data = pd.read_csv(data_path)

    # Standardizing continuous numerical features
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])

    # Converting to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

    # Combining with the original dataset
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # Identifying categorical columns
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  # Exclude target column

    # Applying one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

    # Converting to a DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Combining with the original dataset
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Encoding the target variable
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

    # Splitting data
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # OvO
    model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
    model_ovo.fit(X_train, y_train)

    # OvA
    model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
    model_ova.fit(X_train, y_train)

    # Predictions
    y_pred_ova = model_ova.predict(X_test)
    y_pred_ovo = model_ovo.predict(X_test)

    # Evaluation metrics for OvA
    print("One-vs-All (OvA) Strategy")
    print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

    # Evaluation metrics for OvO
    print("One-vs-All (OvO) Strategy")
    print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")


# %%
obesity_risk_pipeline(file_path, test_size=0.2)

# %%
