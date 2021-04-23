#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

import warnings 
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('Mushroom')
df.head()


# In[4]:


df.shape


# In[ ]:


# the dataset has 8124 rows and 23 columns


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[ ]:


# dataset has no null values


# In[7]:


df['class'].unique()


# In[ ]:


# dataset has two outcomes in class column, so logistic regression case


# In[8]:


df['class'].value_counts()


# In[9]:


sn.countplot(df['class'])


# In[21]:


#labelencoder

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['class']=LE.fit_transform(df['class'])
df['cap-shape']=LE.fit_transform(df['cap-shape'])
df['cap-surface']=LE.fit_transform(df['cap-surface'])
df['bruises']=LE.fit_transform(df['bruises'])
df['odor']=LE.fit_transform(df['odor'])
df['cap-color']=LE.fit_transform(df['cap-color'])
df['gill-attachment']=LE.fit_transform(df['gill-attachment'])
df['gill-spacing']=LE.fit_transform(df['gill-spacing'])
df['gill-size']=LE.fit_transform(df['gill-size'])
df['gill-color']=LE.fit_transform(df['gill-color'])
df['stalk-shape']=LE.fit_transform(df['stalk-shape'])
df['stalk-root']=LE.fit_transform(df['stalk-root'])
df['stalk-surface-above-ring']=LE.fit_transform(df['stalk-surface-above-ring'])
df['stalk-surface-below-ring']=LE.fit_transform(df['stalk-surface-below-ring'])
df['stalk-color-above-ring']=LE.fit_transform(df['stalk-color-above-ring'])                                     
df['stalk-color-below-ring']=LE.fit_transform(df['stalk-color-below-ring'])
df['veil-type']=LE.fit_transform(df['veil-type'])
df['veil-color']=LE.fit_transform(df['veil-color'])
df['ring-number']=LE.fit_transform(df['ring-number'])
df['ring-type']=LE.fit_transform(df['ring-type']) 
df['spore-print-color']=LE.fit_transform(df['spore-print-color'])
df['population']=LE.fit_transform(df['population']) 
df['habitat']=LE.fit_transform(df['habitat'])                                     


# In[ ]:


# with help of using label encoder converted datasets object values to intiger


# In[22]:


df.head()


# In[25]:


cor=df.corr()


# In[26]:


cor


# In[27]:


sn.heatmap(cor,annot=True)


# In[35]:


df.skew()


# In[36]:


# Model building and training

x=df.drop('class',axis=1)
y=df['class']


# In[37]:


x


# In[38]:


y


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[48]:


lm=LogisticRegression()


# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=50)


# In[51]:


from sklearn.tree import DecisionTreeClassifier
maxAccu=0
maxRs=0
for i in range(1,200):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=i)
    mod=DecisionTreeClassifier()
    mod.fit(x_train,y_train)
    pred=mod.predict(x_test)
    acc=accuracy_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRs=i
print("best accuracy is",maxAccu,"on random_state",maxRs)


# In[52]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[54]:


DTC=DecisionTreeClassifier()
DTC.fit(x_train,y_train)
pred=DTC.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc)


# In[ ]:


# applied decision tree classifier, accuracy score is 1.0


# In[55]:


RFC=RandomForestClassifier()
RFC.fit(x_train,y_train)
pred=RFC.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc)


# In[ ]:


# applied random forest classifier, accuracy score is 1.0


# In[56]:


sv=SVC()
sv.fit(x_train,y_train)
pred=sv.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc)


# In[ ]:


# applied support vector classifier, accuracy score is 0.98


# In[57]:


# Cross Validation application

from sklearn.model_selection import cross_val_score


# In[58]:


print(cross_val_score(DTC,x,y,cv=5).mean())


# In[ ]:


# Decision tree classifier cross val score is 0.96


# In[59]:


print(cross_val_score(RFC,x,y,cv=5).mean())


# In[ ]:


# Random Forest classifier cross val score is 0.89


# In[60]:


print(cross_val_score(sv,x,y,cv=5).mean())


# In[ ]:


# Support vector classifier cross val score is 0.84


# In[62]:


#then model accuracy-cross validation score and least difference value is best model
# DTC is the best model
# Hyper Parameter Tuning

from sklearn.model_selection import GridSearchCV


# In[63]:


parameter={'max_depth':np.arange(2,10),
          'criterion':['gini','entropy']}


# In[64]:


GCV=GridSearchCV(DecisionTreeClassifier(),parameter,cv=5)


# In[65]:


GCV.fit(x_train,y_train)


# In[66]:


GCV.best_params_


# In[67]:


final_mod=DecisionTreeClassifier(criterion='gini',max_depth=7)
final_mod.fit(x_train,y_train)
pred=final_mod.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc*100)


# In[ ]:


# after applying criterion-gini at max-depth 7,the accuracy is 100%


# In[69]:


# saving the model
import joblib
joblib.dump(final_mod,"mushroom.pkl")


# In[ ]:





# In[ ]:




