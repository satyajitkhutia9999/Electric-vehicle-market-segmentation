#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


data1 = pd.read_csv("C:\\Users\\satya\\OneDrive\\Documents\\Fynn lab internship\\FINAL TASK\car data.csv")
data1.info()


# In[4]:


import pandas as pd
# Replace 'data' with your DataFrame and 'nominal_columns' with the list of nominal column names
nominal_columns = ['Car_Name', 'Fuel_Type','Seller_Type','Transmission']

# Use Pandas factorize function to perform nominal encoding
for col in nominal_columns:
    data1[col] = pd.factorize(data1[col])[0]
   


# In[5]:


# ELBOW METHOD to get appropriate k value
cluster_range = range(1, 10)  # For example, test clusters from 1 to 10

# Calculate the sum of squared distances (inertia) for different cluster numbers
inertia = []
for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(data1)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method')
plt.show()


# In[6]:


#silhouette_score
from sklearn.metrics import silhouette_score
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data1)
silhouette_avg = silhouette_score(data1, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)


# In[7]:


#silhouette_score
from sklearn.metrics import silhouette_score
n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data1)
silhouette_avg = silhouette_score(data1, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)


# In[8]:


#silhouette_score
from sklearn.metrics import silhouette_score
n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data1)
silhouette_avg = silhouette_score(data1, kmeans.labels_)
print("Silhouette Score:", silhouette_avg)


# In[9]:


kmeans.labels_


# In[10]:


data = pd.read_csv("C:\\Users\\satya\\OneDrive\\Documents\\Fynn lab internship\\FINAL TASK\car data.csv")


# In[11]:


data['Cluster'] = kmeans.labels_


# In[25]:


cluster_0_data = data[data['Cluster'] == 0]
cluster_1_data = data[data['Cluster'] == 1]
cluster_2_data = data[data['Cluster'] == 2]
cluster_3_data = data[data['Cluster'] == 3]
cluster_4_data = data[data['Cluster'] == 4]


# In[14]:


plt.figure(figsize=(20,5))
sns.countplot(data = data, hue = 'Cluster',x='Seller_Type')


# In[15]:


plt.figure(figsize=(20,5))
sns.countplot(data = data, hue = 'Cluster',x='Fuel_Type')


# In[16]:


plt.figure(figsize=(20,5))
sns.countplot(data = data, hue = 'Cluster',x='Year')


# In[27]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_3_data,x='Fuel_Type',y='Selling_Price')


# In[28]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_3_data,x='Seller_Type',y='Selling_Price')


# In[29]:


plt.figure(figsize=(10,5))
sns.barplot(data = cluster_3_data,x='Transmission',y='Selling_Price')


# In[ ]:




