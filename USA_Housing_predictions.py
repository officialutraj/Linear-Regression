
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv('USA_Housing.csv')


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.coloumns()


# In[11]:


df.coloumn


# In[12]:


df.columns


# In[13]:


sns.pairplot(df)


# In[15]:


sns.distplot(df['Price'])


# In[16]:


sns.heatmap(df.corr())


# In[17]:


sns.heatmap(df.corr(), annot = True)


# In[18]:


df.columns


# In[19]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


lm =  LinearRegression()


# In[24]:


lm.fit(X_train,y_train)


# In[25]:


print(lm.intercept_)


# In[30]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])


# In[32]:


coeff_df


# In[33]:


predictions = lm.predict(X_test)
predictions
y_test

# In[37]:


plt.scatter(y_test,predictions)


# In[38]:


sns.distplot((y_test-predictions),bins=50);

