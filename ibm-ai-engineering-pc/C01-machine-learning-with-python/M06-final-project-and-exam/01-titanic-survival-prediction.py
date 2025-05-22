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
# # Practice Project: Titanic Survival Prediction
#

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# %% [markdown]
# ## Load the Titanic dataset using Seaborn

# %%
titanic = sns.load_dataset('titanic')
titanic.head()

# %% [markdown]
# | Variable   |	Definition   |
#  |:------|:--------------------------------|
#  |survived | survived? 0 = No, 1 = yes  |
#  |pclass | Ticket class (int)  |
#  |sex	 |sex |
#  |age	 | age in years  |
#  |sibsp  |	# of siblings / spouses aboard the Titanic |
#  |parch  |	# of parents / children aboard the Titanic |
#  |fare   |	Passenger fare   |
#  |embarked | Port of Embarkation |
#  |class  |Ticket class (obj)   |
#  |who    | man, woman, or child  |
#  |adult_male | True/False |
#  |alive  | yes/no  |
#  |alone  | yes/no  |

# %%
titanic.count()

# %% [markdown]
# `deck` has a lot of missing values so we'll drop it. `age` has quite a few missing values as well. Although it could be, `embarked` and `embark_town` don't seem relevant so we'll drop them as well. It's unclear what `alive` refers to so we'll ignore it.
#
# `survived` is our target class variable.

# %%
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'

X = titanic[features]
y = titanic[target]

# %% [markdown]
# ### Exercise 1. How balanced are the classes?
#

# %%
y.value_counts()

# %% [markdown]
# There's a slight imbalance, we should stratify the data when splitting the train/test set and use cross-validation.

# %% [markdown]
# ### Exercise 2. Split the data into training and testing sets
#

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% [markdown]
# ### Define preprocessing transformers for numerical and categorical features
#
#

# %% [markdown]
# #### Automatically detect numerical and categorical columns and assign them to separate numeric and categorical features

# %%
numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# %% [markdown]
# #### Define separate preprocessing pipelines for both feature types
#

# %%
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# %% [markdown]
# #### Combine the transformers into a single column transformer
#

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# %% [markdown]
# ### Create a model pipeline
#

# %%
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# %% [markdown]
# ### Define a parameter grid 
#

# %%
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# %% [markdown]
# ### Perform grid search cross-validation and fit the best model to the training data
#

# %%
# Cross-validation method
cv = StratifiedKFold(n_splits=5, shuffle=True)

# %% [markdown]
# ### Exercise 3. Train the pipeline model 
#

# %%
model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
model.fit(X_train, y_train)

# %% [markdown]
# ### Exercise 4. Get the model predictions from the grid search estimator on the unseen data
#

# %%
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Exercise 5. Plot the confusion matrix 
#

# %%
# Enter your code here:
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Feature importances
#
# To obtain the categorical feature importances, we have to work our way backward through the modelling pipeline to associate the feature importances with their one-hot encoded input features that were transformed from the original categorical features.

# %%
model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

# %%
feature_importances = model.best_estimator_['classifier'].feature_importances_

# Combine the numerical and one-hot encoded categorical feature names
feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

# %% [markdown]
# ### Display the feature importances in a bar plot
#

# %%
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis() 
plt.title('Most Important Features in predicting whether a passenger survived')
plt.xlabel('Importance Score')
plt.show()

# Print test score 
test_score = model.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")

# %% [markdown]
# ### Exercise 6. These are interesting results to consider. 
# What can you say about these feature importances? Are they informative as is?
#

# %% [markdown]
# Acurracy greater than 80% is satisfactory.
#
# Regarding feature importances, the results are not informative as is, we need to figure out how dependent they are on each other.

# %% [markdown]
# ## Try another model
#
# In practice we want to try out different models and even revisit the data analysis to improve our model performance.
#
# Let's update the pipeline and the parameter grid so we can train a Logistic Regression model and compare the performance of the two models.

# %%
# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# update the model's estimator to use the new pipeline
model.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

model.param_grid = param_grid

# Fit the updated pipeline with Logistic Regression
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# %% [markdown]
# ### Exercise 7. Display the clasification report for the new model and compare the results to your previous model.
#

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# The results are fairly similar, the differences are insignificant.

# %% [markdown]
# ### Exercise 8. Display the confusion matrix for the new model and compare the results to your previous model.
#

# %%
# Enter your code here:
# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()

# What changed in the numbers of true positives and true negatives?

# %% [markdown]
# ### Extract the logistic regression feature coefficients and plot their magnitude in a bar chart.
#

# %%
coefficients = model.best_estimator_.named_steps['classifier'].coef_[0]

# Combine numerical and categorical feature names
numerical_feature_names = numerical_features
categorical_feature_names = (model.best_estimator_.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(categorical_features)
                            )
feature_names = numerical_feature_names + list(categorical_feature_names)

# %% [markdown]
# ### Exercise 9. Plot the feature coefficient magnitudes in a bar chart
# What's different about this chart than the feature importance chart for the Random Forest classifier?
#

# %%
# Create a DataFrame for the coefficients
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs(), color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Coefficient magnitudes for Logistic Regression model')
plt.xlabel('Coefficient Magnitude')
plt.show()

# Print test score
test_score = model.best_estimator_.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")

# %% [markdown]
# The accuracy for both models is pretty much the same. 
#
# However, the features that are important for each model are different. This suggests more work is needed to understand the relationships between the features and the target variable, e.g. correlation, scatter plot, etc.

# %%
