#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Import Libraries and Load Data##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
#df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your file path
df = pd.read_csv (r'/Users/ballu_macbookpro/Downloads/titanic.csv', sep=',')


# In[2]:


#Basic Dataset Overview#
print("Shape of the dataset:", df.shape)
print("Data types:\n", df.dtypes)



# In[6]:


##Display First Few Rows##
df.head()


# In[7]:


Check for Missing Values

print("Missing values per column:\n", df.isnull().sum())


# In[10]:


##Summary Statistics##

df.describe(include='all')  # Shows statistics for both numerical and categorical columns


# In[11]:


#Univariate Analysis#

##Distribution of Numerical Features##

numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[12]:


#Distribution of Categorical Features#

categorical_columns = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Count plot of {col}')
    plt.xticks(rotation=45)
    plt.show()


# In[15]:


#Bivariate Analysis#

##Numerical vs Numerical (Correlation Heatmap)##

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


# In[16]:


#Categorical vs Numerical (Box Plot)#

for cat_col in categorical_columns:
    for num_col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f'{num_col} by {cat_col}')
        plt.xticks(rotation=45)
        plt.show()


# In[ ]:


#Outlier Detection#

##Box Plot for Outliers##

for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Box plot of {col}')
    plt.show()


# In[17]:


#Checking for Skewness#

skew_values = df[numeric_columns].skew()
print("Skewness of numerical features:\n", skew_values)


# In[18]:


#Multivariate Analysis (Pair Plot)#
sns.pairplot(df[numeric_columns])
plt.show()


# In[ ]:




