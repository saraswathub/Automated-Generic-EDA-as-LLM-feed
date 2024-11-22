#!/usr/bin/env python
# coding: utf-8




##Import Libraries and Load Data##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
#df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your file path
df = pd.read_csv (r'/Users/xxxx/Downloads/titanic.csv', sep=',')





#Basic Dataset Overview#
print("Shape of the dataset:", df.shape)
print("Data types:\n", df.dtypes)





##Display First Few Rows##
df.head()





Check for Missing Values

print("Missing values per column:\n", df.isnull().sum())




##Summary Statistics##

df.describe(include='all')  # Shows statistics for both numerical and categorical columns




#Univariate Analysis#

##Distribution of Numerical Features##

numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()





#Distribution of Categorical Features#

categorical_columns = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Count plot of {col}')
    plt.xticks(rotation=45)
    plt.show()


#Bivariate Analysis#

##Numerical vs Numerical (Correlation Heatmap)##

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()




#Categorical vs Numerical (Box Plot)#

for cat_col in categorical_columns:
    for num_col in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=cat_col, y=num_col, data=df)
        plt.title(f'{num_col} by {cat_col}')
        plt.xticks(rotation=45)
        plt.show()





#Outlier Detection#

##Box Plot for Outliers##

for col in numeric_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Box plot of {col}')
    plt.show()



#Checking for Skewness#

skew_values = df[numeric_columns].skew()
print("Skewness of numerical features:\n", skew_values)




#Multivariate Analysis (Pair Plot)#
sns.pairplot(df[numeric_columns])
plt.show()







