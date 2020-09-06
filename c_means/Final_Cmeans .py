from fcmeans import FCM

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




df = pd.read_excel (r'C://data.xlsx')

df1 = pd.read_excel (r'C://data.xlsx')

print(df.keys())

del df['Name']

del df['Roll NO']


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


####Select area by adding in Area_choice

Area_choice=[s1,s2,s12,s14,s15]

#Area_choice=[s1]

#Area_choice=[s1,s2,s12,s14,s15,s7,s9]



C_means_model_1 = FCM(n_clusters=5,random_state=123)

distances_1 = C_means_model_1.fit(df[Area_choice])

labels_1 = C_means_model_1.u.argmax(axis=1)

df['cluster_1']=labels_1

df['cluster_1_label']=df['cluster_1'].apply(cluster_1_label)


Index_label = df[df['cluster_1_label']==3].index.tolist()

Cluster = Index_label
Index_label = np.array(Index_label)

t = 1

temp_df_Roll0 = df1.ix[(df['cluster_1'] == 0), ['Roll NO','Name']]
label0=np.full((1,len(temp_df_Roll0)),0)
temp_df_Data0 = df.ix[(df['cluster_1'] == 0), Area_choice]
print('CMEANS Students-Weak ', Area_choice)
print(temp_df_Data0)
temp_df_Roll1 = df1.ix[(df['cluster_1'] == 1), ['Roll NO','Name']]
label1=np.full((1,len(temp_df_Roll1)),1)
temp_df_Data1 = df.ix[(df['cluster_1'] == 1), Area_choice]
print('CMEANS Students-Third Category ', Area_choice)
print(temp_df_Data1)
temp_df_Roll2 = df1.ix[(df['cluster_1'] == 2), ['Roll NO','Name']]
label2=np.full((1,len(temp_df_Roll2)),2)
temp_df_Data2 = df.ix[(df['cluster_1'] == 2), Area_choice]
print('CMEANS Students-Second class', Area_choice)
print(temp_df_Data2)
temp_df_Roll3 = df1.ix[(df['cluster_1'] == 3), ['Roll NO','Name']]
label3=np.full((1,len(temp_df_Roll3)),3)
temp_df_Data3 = df.ix[(df['cluster_1'] == 3), Area_choice]
print('CMEANS Students-First class ', Area_choice)
print(temp_df_Data3)
temp_df_Roll4 = df1.ix[(df['cluster_1'] == 4), ['Roll NO','Name']]
label4=np.full((1,len(temp_df_Roll4)),4)
temp_df_Data4 = df.ix[(df['cluster_1'] == 4), Area_choice]
print('CMEANS Students-Distinction', Area_choice)
print(temp_df_Data4)
hst0=len(temp_df_Data0)
hst1=len(temp_df_Data1)
hst2=len(temp_df_Data2)
hst3=len(temp_df_Data3)
hst4=len(temp_df_Data4)

hist=[]
hist.append(hst0)
hist.append(hst1)
hist.append(hst2)
hist.append(hst3)
hist.append(hst4)
print(hist)
axisx=[]
for i in range(0,5):
	axisx.append(i)

###HISTOGRAM PLOT
plt.bar(axisx,hist)
plt.ylabel('No of Students')
plt.xlabel('label')
plt.show()

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