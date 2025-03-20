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
# # Decision Trees

# %%
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics

# %matplotlib inline

import warnings
warnings.filterwarnings('ignore')

# %%
path = './drug200.csv'

# %%
my_data = pd.read_csv(path)
my_data

# %% [markdown]
# ## Data Analysis and pre-processing

# %%
my_data.info()

# %% [markdown]
# This tells us that 4 out of the 6 features of this dataset are categorical, which will have to be converted into numerical ones to be used for modeling. For this, we can make use of __LabelEncoder__ from the Scikit-Learn library.
#

# %%
label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex']) 
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol']) 
my_data

# %% [markdown]
# With this, we now have 5 parameters that can be used for modeling and 1 feature as the target variable. 
# We can see from comparison of the data before Label encoding and after it, to note the following mapping.
# <br>
# For parameter 'Sex' : $M \rightarrow 1, F \rightarrow 0$ <br>
# For parameter 'BP' : $High \rightarrow 0, Low \rightarrow 1, Normal \rightarrow 2$<br>
# For parameter 'Cholesterol' : $High \rightarrow 0, Normal \rightarrow 1$
#

# %%
my_data.isnull().sum()

# %% [markdown]
# This tells us that there are no missing values in any of the fields.
#

# %% [markdown]
# To evaluate the correlation of the target variable with the input features, it will be convenient to map the different drugs to a numerical value.

# %%
custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}
my_data['Drug_num'] = my_data['Drug'].map(custom_map)
my_data

# %%
my_data.drop('Drug', axis=1).corr()

# %% [markdown]
# We can also understand the distribution of the dataset by plotting the count of the records with each drug recommendation. 
#

# %%
category_counts = my_data['Drug'].value_counts()

# Plot the count plot
plt.bar(category_counts.index, category_counts.values, color='blue')
plt.xlabel('Drug')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)  # Rotate labels for better readability if needed

# %% [markdown]
# ## Modeling
#

# %% [markdown]
# For modeling this dataset with a Decision tree classifier, we first split the dataset into training and testing subsets. For this, we separate the target variable from the input variables.
#

# %%
y = my_data['Drug']
X = my_data.drop(['Drug','Drug_num'], axis=1)

# %%
# 30% of the data will be used for testing
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=32)

# %%
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)

# %% [markdown]
# ### Evaluation
#

# %%
tree_predictions = drugTree.predict(X_testset)
tree_predictions

# %%
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

# %% [markdown]
# ### Visualize the tree
#

# %%
plot_tree(drugTree)

# %% [markdown]
# From this tree, we can derive the criteria developed by the model to identify the class of each training sample. We can interpret them by tracing the criteria defined by tracing down from the root to the tree's leaf nodes.
#
# For instance, the decision criterion for Drug Y is ${Na\_to\_K} \gt 14.627$.
#

# %% [markdown]
# #### Practice Question:
#
# If the max depth of the tree is reduced to 3, how would the performance of the model be affected?
#

# %%
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 3)
drugTree.fit(X_trainset,y_trainset)

tree_predictions = drugTree.predict(X_testset)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_testset, tree_predictions))

# %%
