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
# # K-means Clustering
#
# **K-means** is vastly used for clustering in many data science applications. It is especially useful if you need to quickly discover insights from **unlabeled data**.

# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# %% [markdown]
# ## K-Means on a synthetic data set

# %%
np.random.seed(0)

# %%
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# %%
plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.3,ec='k',s=80)

# %% [markdown]
# ## Setting up k-means

# %%
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# %%
k_means.fit(X)

# %%
k_means_labels = k_means.labels_
k_means_labels

# %%
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

# %% [markdown]
# ## Creating the Visual Plot

# %%
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# %% [markdown]
# ### Exercise 1
# Try to cluster the above dataset into a different number of clusters, say k=3. Note the difference in the pattern generated.
#

# %%
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)

k_means.fit(X)

k_means_labels = k_means.labels_

k_means_cluster_centers = k_means.cluster_centers_

# %%
fig = plt.figure(figsize=(6, 4))

colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1)

for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = (k_means_labels == k)

    cluster_center = k_means_cluster_centers[k]

    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)


# %% [markdown]
# ## Customer Segmentation with k-means

# %%
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
cust_df

# %% [markdown]
# ### Pre-processing

# %% [markdown]
# `Address` in this dataset is a categorical variable. The k-means algorithm doesn't work directly with categorical variables because the Euclidean distance function isn't meaningful for them. You could one-hot encode the categorical feature, but for illustration purposes let's run k-means clustering without it.

# %%
cust_df = cust_df.drop('Address', axis=1)

# %%
cust_df = cust_df.dropna()
cust_df.info()

# %% [markdown]
# #### Normalizing over the standard deviation
#
# Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally, tranforming the features so they have zero mean and standard deviation of one.

# %%
X = cust_df.values[:,1:] # leaves out `Customer ID`
Clus_dataSet = StandardScaler().fit_transform(X)

# %% [markdown]
# ### Modeling

# %% [markdown]
# ### Exercise 4
# Write the code to cluster the data with k=3. Extract the cluster labels for this clustering process.
#

# %%
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means.fit(X)
labels = k_means.labels_

# %% [markdown]
# ### Insights
#
# We assign the k-means cluster labels to each row in the dataframe.
#

# %%
cust_df["Clus_km"] = labels

# %% [markdown]
# We can easily check the centroid values by averaging the features in each cluster. These values indicate the central point of the cluster from the vantage point of the field in question.
#

# %%
cust_df.groupby('Clus_km').mean()

# %% [markdown]
# Distribution of customers based on their education, age and income.

# %%
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k',alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

# %% [markdown]
# We can also see this distribution in 3 dimensions for better understanding.

# %%
# Create interactive 3D scatter plot
fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Remove color bar, resize plot

# %% [markdown]
# k-means will partition our customers into mutually exclusive groups, for example, into 3 clusters. The customers in each cluster are similar to each other demographically.
