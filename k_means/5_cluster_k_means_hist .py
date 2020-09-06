#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


# In[2]:


df = pd.read_excel (r'C:\\data.xlsx')

df1 = pd.read_excel (r'C:\\data.xlsx')


# In[3]:


print(df.keys())

del df['Name']

del df['Roll NO']


# In[4]:


def cluster_1_label(alpha):

    if alpha == 0:

        return 'Weak'

    if alpha == 1:

        return 'Third'

    if alpha == 2:

        return 'Second'

    if alpha == 3:

        return 'First'

    if alpha == 4:

        return 'Distinction'


# In[5]:


s1='SSC Marks'

s2='HSC Marks'

s3='cpga'

s4='communication'

s5='aptitude'

s6='Leadership'

s7='database'

s8='networking'

s9='os'

s10='Software Testing'

s11='data structure'

s12='python'

s13='android'

s14='C'

s15='C++'

s16='java'


# In[6]:


Area_choice=[s1,s2,s12,s14,s15]


# In[7]:


kmeans_model_1 = KMeans(n_clusters=5,random_state=123)

distances=kmeans_model_1.fit_transform(df[Area_choice])
kmeans_model_1.fit(df[Area_choice])
centers=kmeans_model_1.cluster_centers_
print('Centers=')


# In[8]:


print(centers)
labels_1 = kmeans_model_1.predict(df[Area_choice])
df['cluster_1']=labels_1

df['cluster_1_label']=df['cluster_1'].apply(cluster_1_label)


# In[9]:


Index_label = df[df['cluster_1_label']==3].index.tolist()


# In[10]:


Cluster = Index_label
Index_label = np.array(Index_label)


# In[11]:


t = 1


# In[18]:



temp_df_Data0 = df1.ix[(df['cluster_1'] == 0), ['Roll NO','Name']]
temp_df_Data1=df1.ix[(df['cluster_1'] == 1), ['Roll NO','Name']]
temp_df_Data2=df1.ix[(df['cluster_1'] == 2), ['Roll NO','Name']]
temp_df_Data3=df1.ix[(df['cluster_1'] == 3), ['Roll NO','Name']]
temp_df_Data4=df1.ix[(df['cluster_1'] == 4), ['Roll NO','Name']]

print("    ")
print("weak students : ")
print(temp_df_Data0)
print("    ")
print("third students : ")
print("    ")
print(temp_df_Data1)
print("    ")
print("second students : ")
print("    ")
print(temp_df_Data2)
print("    ")
print("    ")
print("first students : ")
print(temp_df_Data3)
print("    ")
print("distinction students : ")
print(temp_df_Data4)


# In[19]:


hst0=len(temp_df_Data0)
hst1=len(temp_df_Data1)
hst2=len(temp_df_Data2)
hst3=len(temp_df_Data3)
hst4=len(temp_df_Data4)


# In[20]:


hist=[]
hist.append(hst0)
hist.append(hst1)
hist.append(hst2)
hist.append(hst3)
hist.append(hst4)
print(hist)


# In[21]:


axisx=[]
for i in range(0,5):
	axisx.append(i)


# In[22]:


plt.bar(axisx,hist)
plt.ylabel('No of Students')
plt.xlabel('label')
plt.show()


# In[17]:


df.sort_values(by=['cluster_1'], inplace=True)
X=range(0,len(df))
Y=df[Area_choice]
for i in range(0,len(Area_choice)):
	plt.scatter(temp_df_Data0[Area_choice[i]], label0, c='yellow', s=300, cmap='viridis')
	plt.scatter(temp_df_Data1[Area_choice[i]], label1, c='red', s=300)
	plt.scatter(temp_df_Data2[Area_choice[i]], label2, c='blue', s=300)
	plt.scatter(temp_df_Data3[Area_choice[i]], label3, c='green', s=300)
	plt.scatter(temp_df_Data4[Area_choice[i]], label4, c='black', s=300)
	plt.ylabel('Cluster ID')
	plt.xlabel(Area_choice[i])
	plt.show()


# In[ ]:




