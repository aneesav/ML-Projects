#!/usr/bin/env python
# coding: utf-8

# # Data Loading

# ## Data loading with Pandas
# - Load `adult` dataset using Pandas.
# - Check the first five observations

# In[5]:


import pandas as pd
adult = pd.read_csv("/Users/aneesavalentine/Downloads/adult.csv", index_col = 0)
adult.head(5)

### Index col function creates an indexed column with a number corresponding to each observation.


# ## Simple analysis
# - Check the shape of the dataset, 
# - list the column names count number of unique values in `income` column, 
# - plot the histogram of age grouped by income

# In[6]:


adult.shape

### Format of data.shape output = (observations, features or rows, columns).


# In[9]:


list(adult.columns)

### We assume the income column is the target variable so we want to know the names of all the other columns.
### List function is optional.


# In[11]:


adult.income

### Shows first and last values in the long "income" vector.
### We are just doing data exploration now.


# In[13]:


adult.income.value_counts()

### Value_counts counts the number of values in the "income" column. We have 24,000 datapoints that are less than 50k abd 7000+ that are more than 50k
### Also tells us the data is unbalanced.


# In[14]:


adult.groupby(["income"])

### Returns an object with two groups: one with >50k and the other <50k.


adult.groupby(["income"]).age.hist()

### Returns a histogram demonstrating tradeoff between age and income using our previously generated object.
### Bars = income since there are only 2 groups - 2 colors represent those groups. High income = blue, low income = 


# # Splitting into training and test data
# - Assign all the data features to X
# - Assign the target variable to y
# - Split data into train and test datasets

# In[ ]:





# # Exercise I 
# Load the "boston house prices" dataset from the ``boston_house_prices.csv`` file using the ``pd.read_csv`` function (you don't need ``index_column`` here).
# You can find a description of this dataset in the ``boston_house_prices.txt`` file.
# 
# This is a regression dataset with "MEDV" the median house value in a block in thousand dollars the target.
# How many features are there and how many samples?
# 
# Split the data into a training and a test set for learning.
# Optionally you can plot MEDV vs any of the features using the ``plot`` method of the dataframe (using ``kind="scatter"``).

# In[ ]:





# ## Load Datasets from ScikitLearn
# Load digits dataset from sklearn

# In[ ]:





# # Exercise II
# 
# Load the iris dataset from the ``sklearn.datasets`` module using the ``load_iris`` function.
# The function returns a dictionary-like object that has the same attributes as ``digits``.
# 
# What is the number of classes, features and data points in this dataset?
# Use a scatterplot to visualize the dataset.
# 
# You can look at ``DESCR`` attribute to learn more about the dataset.
# 

# In[ ]:




