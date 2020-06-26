#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv(r"C:\Users\nived\opencv\KPMG2.csv",index_col=0)
df.head()


# In[4]:


plt.figure(figsize=(10,6))
plt.title("Ages Frequency")
sns.axes_style("dark")
sns.violinplot(y=df["age"])
plt.show()


# In[5]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.boxplot(y=df["tenure"], color="red")
plt.subplot(1,2,2)
sns.boxplot(y=df["past_3_years_bike_related_purchases"])
plt.show()


# In[6]:


genders = df.gender.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=genders.index, y=genders.values)
plt.show()


# In[7]:



age18_25 = df.age[(df.age <= 25) & (df.age >= 18)]
age26_35 = df.age[(df.age <= 35) & (df.age >= 26)]
age36_45 = df.age[(df.age <= 45) & (df.age >= 36)]
age46_55 = df.age[(df.age <= 55) & (df.age >= 46)]
age55above = df.age[df.age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age18_25.values),len(age26_35.values),len(age36_45.values),len(age46_55.values),len(age55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()


# In[23]:


y = df['past_3_years_bike_related_purchases']
X = df[['property_valuation','age']]


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


lm = LinearRegression()

lm1= lm.fit(X_train,y_train)

import statsmodels.api as sm
from scipy import stats
X2 = sm.add_constant(X_test)
est = sm.OLS(y_test, X2)
est2 = est.fit()
print(est2.summary())


# In[27]:


print('Coefficients: \n', lm.coef_)

