#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[14]:


df=pd.read_csv('mall_customers.csv')
print(df)


# In[15]:


plt.scatter(df.iloc[:,-2],df.iloc[:,-1])
plt.xlabel("Annual income")
plt.ylabel("Spending score")                 
plt.show()




# In[ ]:


X=df.iloc[:,-2:]


# In[16]:


# elbow method use for identity how many cluster should make

wcss=[]                                                                                          
from sklearn.cluster import KMeans  

for i in range(1,11):
    kmeans=KMeans(n_clusters=i) 
    kmeans.fit(X)  
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss) 
plt.xlabel("No of cluster")
plt.ylabel("WCSS")
plt.show()


# In[17]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
labels=kmeans.fit_predict(X) 
print(labels) 


# In[18]:


plt.scatter(X.loc[labels==0,"Annual Income (k$)"],X.loc[ labels==0,"Spending Score (1-100)"])
plt.scatter(X.loc[labels==1,"Annual Income (k$)"],X.loc[ labels==1,"Spending Score (1-100)"],color="red")
plt.scatter(X.loc[labels==2,"Annual Income (k$)"],X.loc[ labels==2,"Spending Score (1-100)"],color="yellow")
plt.scatter(X.loc[labels==3,"Annual Income (k$)"],X.loc[ labels==3,"Spending Score (1-100)"],color="blue")
plt.scatter(X.loc[labels==4,"Annual Income (k$)"],X.loc[ labels==4,"Spending Score (1-100)"],color="green")
plt.show()  


# In[19]:


from scipy.cluster import hierarchy as sch
sch.dendrogram(sch.linkage(X,method='ward'))   # use for making denrogram to find the require numbers of clustering
plt.xlabel('points')
plt.ylabel('distance')
plt.title('dendrogram')
plt.show()


# In[21]:


from sklearn.cluster import AgglomerativeClustering
agg=AgglomerativeClustering(n_clusters=5)

labels=agg.fit_predict(X)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




