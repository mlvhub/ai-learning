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
#  # Principal Component Analysis (PCA)

# %% [markdown]
# ## Part I: Using PCA to project 2-D data onto its principal axes

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# %%
# create a 2-dimensional dataset containing two linearly correlated features
np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)
X

# %% [markdown]
# ### Exercise 1. Visualize the relationship between the two features.
#

# %%
plt.figure()
plt.scatter(X[:,0], X[:,1], edgecolor='k', alpha=0.7)
plt.title("Scatter Plot of Bivariate Normal Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.grid(True)

# %% [markdown]
# ### Perform PCA on the dataset

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# %% [markdown]
# ### Get the principal components from the model

# %%
components = pca.components_
components

# %%
# the principal components are sorted in decreasing order by their explained variance, 
# which can be expressed as a ratio
pca.explained_variance_ratio_

# %% [markdown]
# ### Exercise 2. What percentage of the variance in the data is explained by the first principal component?
#

# %% [markdown]
# A. The first component explains 91.12% of the variance in the data, the second component explains 8.88% of the variance.

# %% [markdown]
# ### Display the results
#

# %%
projection_pc1 = np.dot(X, components[0])
projection_pc2 = np.dot(X, components[1])

# %%
x_pc1 = projection_pc1 * components[0][0]
y_pc1 = projection_pc1 * components[0][1]
x_pc2 = projection_pc2 * components[1][0]
y_pc2 = projection_pc2 * components[1][1]

# %%
# Plot original data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], label='Original Data', ec='k', s=50, alpha=0.6)

# Plot the projections along PC1 and PC2
plt.scatter(x_pc1, y_pc1, c='r', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 1')
plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 2')
plt.title('Linearly Correlated Data Projected onto Principal Components', )
plt.xlabel('Feature 1',)
plt.ylabel('Feature 2',)
plt.legend()
plt.grid(True)
plt.axis('equal')

# %% [markdown]
# ### Exercise 3. Describe the second direction.
#

# %% [markdown]
# The second direction is perpendicular to the first and has a lower variance.

# %% [markdown]
# ## Part II. PCA for feature space dimensionality reduction

# %%
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# ### Exercise 4. What are the Iris flower's names?
#

# %%
iris.target_names

# %% [markdown]
# ### Exercise 5. Initialize a PCA model and reduce the Iris data set dimensionality to two components
#

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

components = pca.components_
components

# %%
pca.explained_variance_ratio_

# %%
# Plot the PCA-transformed data in 2D
plt.figure(figsize=(8,6))

colors = ['navy', 'turquoise', 'darkorange']
lw = 1

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=50, ec='k',alpha=0.7, lw=lw,
                label=target_name)

plt.title('PCA 2-dimensional reduction of IRIS dataset',)
plt.xlabel("PC1",)
plt.ylabel("PC2",)
plt.legend(loc='best', shadow=False, scatterpoints=1,)

# %% [markdown]
# ### Exercise 6. What percentage of the original feature space variance do these two combined principal components explain?
#

# %%
pca.explained_variance_ratio_.sum() * 100

# %% [markdown]
# ## A deeper look at the explained variances
#

# %% [markdown]
# ### Exercise 7. Reinitialize the PCA model without reducing the dimension
#

# %%
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

components = pca.components_
components

# %%
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio

# %%
explained_variance_ratio.sum() * 100

# %%
# Plot explained variance ratio for each component
plt.figure(figsize=(10,6))
plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio, alpha=1, align='center', label='PC explained variance ratio' )
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance by Principal Components')

# Plot cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.step(range(1, 5), cumulative_variance, where='mid', linestyle='--', lw=3,color='red', label='Cumulative Explained Variance')
# Only display integer ticks on the x-axis
plt.xticks(range(1, 5))
plt.legend()
plt.grid(True)

# %%
