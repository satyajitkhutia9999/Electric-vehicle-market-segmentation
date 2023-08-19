#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import scipy
import re


# In[5]:


data2 = pd.read_csv("C:\\Users\\satya\\OneDrive\\Documents\\Fynn lab internship\\FINAL TASK\\CAR DETAILS FROM CAR DEKHO.csv")
data2.info()


# In[6]:


import pandas as pd
# Replace 'data' with your DataFrame and 'nominal_columns' with the list of nominal column names
nominal_columns = ['name', 'fuel','seller_type','transmission','owner']

# Use Pandas factorize function to perform nominal encoding
for col in nominal_columns:
    data2[col] = pd.factorize(data2[col])[0]
  


# In[7]:


# ELBOW METHOD to get appropriate k value
cluster_range = range(1, 9)  # For example, test clusters from 1 to 10

# Calculate the sum of squared distances (inertia) for different cluster numbers
inertia = []
for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data2)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method')
plt.show()


# In[8]:


#silhouette_score
from sklearn.metrics import silhouette_score
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data2)
silhouette_avg = silhouette_score(data2, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)


# In[9]:


kmeans.labels_


# In[32]:


data = pd.read_csv("C:\\Users\\satya\\OneDrive\\Documents\\Fynn lab internship\\FINAL TASK\\CAR DETAILS FROM CAR DEKHO.csv")


# In[33]:


data['Cluster'] = kmeans.labels_


# In[34]:


data.tail()


# In[35]:


cluster_0_data = data[data['Cluster'] == 0]
cluster_1_data = data[data['Cluster'] == 1]
cluster_2_data = data[data['Cluster'] == 2]


# In[17]:


plt.figure(figsize=(20,5))
sns.countplot(data = data2, hue = 'Cluster',x='year')


# In[38]:


plt.figure(figsize=(20,5))
sns.countplot(data = data, hue = 'Cluster',x='fuel')


# In[39]:


plt.figure(figsize=(20,5))
sns.countplot(data = data, hue = 'Cluster',x='seller_type')


# In[40]:


plt.figure(figsize=(20,5))
sns.countplot(data = data, hue = 'Cluster',x='transmission')


# In[25]:


plt.figure(figsize=(20,5))
sns.countplot(data = data2, hue = 'Cluster',x='owner')


# In[45]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_0_data,x='fuel',y='selling_price')


# In[46]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_1_data,x='fuel',y='selling_price')


# In[48]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_2_data,x='fuel',y='selling_price')


# In[42]:


cluster_0_data.head()


# In[50]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_0_data,x='owner',y='selling_price')


# In[51]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_2_data,x='owner',y='selling_price')


# In[52]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_1_data,x='owner',y='selling_price')


# In[ ]:




