#!/usr/bin/env python
# coding: utf-8

# # Building an End-to-End Predictive Model 
# ## Fuel Consumption Predictor - Exploratory Data Analysis

# In this project, I will build a Machine Learning application from data acquisition to deployment/production. 
# 
# Using the publicly acquired `Auto MPG Dataset` from the UC Irvine Machine Learning Respository, I will build a regression model to predict city-cycle fuel consumption in miles per gallon (mpg). I will then fine tune the model hyperparameters, perform cross-validation and ultimately deploy the model to Heroku, using Flask.

# In[573]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[574]:


# defining column names
cols = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv('./auto-mpg.data', names=cols, na_values = "?",
                comment = '\t',
                sep= " ",
                skipinitialspace=True)

data = df.copy()


# In[575]:


# taking a look at a few rows in the data, to help define the problem
# the MPG variable is our target variable. MPG = continuous
# our aim is to predict the MPG value for a vehicle, based on the other vehicle attributes/features

data.sample(20)


# # Problem Statement:
# The data contains the MPG variable which is continuous data, and tells us about the efficiency of fuel consumption of a vehicle in 70s and 80s.
# 
# Our aim here is to predict the MPG value for a vehicle given the other attributes of that vehicle.

# Now, we will perform Exploratory Data Analysis on the dataset.

# In[576]:


# checking the data info

data.info()


# In[577]:


# check for null values

data.isnull().sum()


# The horsepower column has 6 missing values, let's look into that a bit more.

# In[578]:


# check for outliers
# summary statistics of quantitative variables

data.describe()


# In[579]:


# looking at horsepower box plot

sns.boxplot(x=data['Horsepower'])


# Since there are a few outliers, we will use the median of the column to impute the missing values. 
# Note: Imputing missing values may not always be best practice, and largely depends on the use case. It might be more appropriate to remove the respective rows altogether.

# In[580]:


# imputing the values with median

median = data['Horsepower'].median()
data['Horsepower'] = data['Horsepower'].fillna(median)
data.info()


# Now, we will assess the category distribution in the categorical columns.

# In[581]:


data["Cylinders"].value_counts() / len(data)


# In[582]:


data['Origin'].value_counts()


# The two categorical columns are "Cylinders" & "Origin". Here, we see that either field only has a few categories of values. Looking at the value distribution will give us insight into the overall data distribution.

# Now, let's look at how each variable might be correlated with the other, using pairplots.

# In[583]:


# pairplots to get an sense of potential correlations

sns.pairplot(data[["MPG", "Cylinders", "Displacement", "Weight", "Horsepower"]], diag_kind="kde")


# Here, we notice that MPG, our target variable, is negatively correlated with displacement, weight & horsepower.

# There is no right way to conduct Exploratory Data Analysis, but now that we have a better understanding of the data distribution and the relationship between variables, we will set aside our test dataset, for validating our final model. This validation should be performed on data that is not used in any other part of the model build. 
# 
# 

# There are many ways to split data - it is certainly not a "one size fits all" endeavor. For this project, we want our test set to be representative of the entire population not only any specific category. So, instead of using sklearn's built in `train_test_split` function, we will leverage stratified sampling to create homogeneous subgroups called strata from the overall population and sample the right number of instances to each stratum to ensure that the test set is representative of the overall population.
# 
# However, we will use the `test_test_split` function first, to compare the split between the two methods.

# In[584]:


# set aside the test data
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

test_set.shape


# In[585]:


train_set['Cylinders'].value_counts() / len(train_set)


# In[586]:


test_set["Cylinders"].value_counts() / len(test_set)


# Now we will use stratified sampling. Since we saw how the categorical data in the "Cylinders" column was distributed prior, we will use that column to create the strata. 

# In[587]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Cylinders"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]


# In[588]:


strat_test_set.shape


# In[589]:


# checking for cylinder category distribution in training set

strat_train_set['Cylinders'].value_counts() / len(strat_train_set)


# In[590]:


# checking for cylinder category distribution in testing set
strat_test_set["Cylinders"].value_counts() / len(strat_test_set)


# In[591]:


# converting integer classes to countries in Origin column
train_set['Origin'] = train_set['Origin'].map({1: 'India', 2: 'USA', 3 : 'Germany'})
train_set.sample(10)


# In[592]:


# one hot encoding
train_set = pd.get_dummies(train_set, prefix='', prefix_sep='')
train_set.head()


# In[593]:


data = strat_train_set.copy()


# Now that we've pre-processed the data up to this point, we're interested in seeing the relationship between each variable and our target variable (MPG).

# In[594]:


# testing new variables by checking their correlation with respect to MPG
corr_matrix = data.corr()
corr_matrix['MPG'].sort_values(ascending=False)


# Let's create some new variables derived from our existing variables, again with respect to MPG.

# In[595]:


# testing new variables by checking their correlation w.r.t. MPG
data['displacement_on_power'] = data['Displacement'] / data['Horsepower']
data['weight_on_cylinder'] = data['Weight'] / data['Cylinders']
data['acceleration_on_power'] = data['Acceleration'] / data['Horsepower']
data['acceleration_on_cyl'] = data['Acceleration'] / data['Cylinders']

corr_matrix = data.corr()
corr_matrix['MPG'].sort_values(ascending=False)


# Interestingly, we see that our two new variables `acceleration_on_power` and `acceleration_on_cyl` are more postitively correlated with MPG than the initial variables.

# Now, this Exploratory Data Analysis is sufficient to move onto the fun part! At this point we've completed ~80% of the model build, now we can move onto the actual Machine Learning. First, we will automate essential parts of our process using the `scikit-learn` package, to make integration into the ML pipeline and final product more efficient.

# To handle missing values, we will use the `SimpleImputer` class from the impute module of the sklearn library. 

# In[596]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(data)


# In[597]:


imputer.statistics_


# In[598]:


data.median().values


# In[599]:


X = imputer.transform(data)


# In[600]:


data_tr = pd.DataFrame(X, columns=data.columns,
                          index=data.index)


# # Part 1 ends here

# In[601]:


# segregating target and feature variables 
data = strat_train_set.drop("MPG", axis=1)
data_labels = strat_train_set["MPG"].copy()
data


# Now, we will automate our preprocessing of the "Origin" column.

# In[602]:


# preprocess the origin (categorical) column
def preprocess_origin_cols(df):
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df
data_tr = preprocess_origin_cols(data)
data_tr.head()


# In[603]:


data_tr.info()


# In[604]:


# isolating the origin column
data_cat = data_tr[["Origin"]]
data_cat.head()


# In[605]:


# onehotencoding the categorical values
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
data_cat_1hot = cat_encoder.fit_transform(data_cat)
data_cat_1hot   # returns a sparse matrix


# In[606]:


data_cat_1hot.toarray()[:5]


# In[607]:


cat_encoder.categories_


# Now, to automate handling missing values using the `SimpleImputer` class.

# In[608]:


# segregating the numerical columns
num_data = data.iloc[:, :-1]
num_data.info()


# In[609]:


# handling missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(num_data)


# In[610]:


# median of all the columns from imputer
imputer.statistics_


# In[611]:


# median from pandas dataframe - same
data.median().values


# In[612]:


# imputing the missing values by transforming the dataframe to an array
X = imputer.transform(num_data)
X


# In[613]:


# converting the 2D array back into a dataframe
data_tr = pd.DataFrame(X, columns=num_data.columns,
                          index=num_data.index)
data_tr.info()


# To make changes to our dataset and create new attributes, we will use the sklearn `BaseEstimator` and `Transformer` classes. We will develop new features by defining our own class. As part of our EDA step above, we calculated two new features:
# 
# - acc_on_power — Acceleration divided by Horsepower
# - acc_on_cyl — Acceleration divided by the number of Cylinders
# 
# We will create a class to add these two features.

# In[614]:


num_data.head()


# In[615]:


from sklearn.base import BaseEstimator, TransformerMixin

acc_ix, hpower_ix, cyl_ix = 4, 2, 0

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): # no *args or **kargs
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        
        return np.c_[X, acc_on_cyl]
    
attr_adder = CustomAttrAdder(acc_on_power=True)
data_tr_extra_attrs = attr_adder.transform(data_tr.values)
data_tr_extra_attrs[0]


# Now, we want to set up a data transformation pipeline to handle both numerical and categorical variables by automating major data transformations. Major transformations will primarily be done on our numerical data, so we will use sklearn's `Pipeline` class to create a numerical pipeline.

# In[616]:


# Using Pipeline class
from sklearn.pipeline import Pipeline
# Using StandardScaler to scale all the numerical attributes
from sklearn.preprocessing import StandardScaler

numerics = ['float64', 'int64']

num_data = data_tr.select_dtypes(include=numerics)

# pipeline for numerical attributes
# imputing -> adding attributes -> scale them
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
    ])

num_data_tr = num_pipeline.fit_transform(num_data)
num_data_tr[0]


# In the above code, we have cascaded a set of 3 transformations:
# 
# 1. Imputing Missing Values, using the `SimpleImputer` class discussed above.
# 2. Custom Attribute Addition, using the custom attribute class defined above.
# 3. Standard Scaling of each Attribute. Note: It is good practice to scale the values before feeding them to the ML model, using the `standardScaler` class.

# Now that we have our numerical transformation ready, we need to configure our categorical data. The only categorical column we have is the "Origin" column. We need to one-hot encode these values. 
# 
# Once we're finished handling the categorical column, we will combine both numerical and categorical pipelines into a single flow using the `ColumnTransformer` class.

# In[617]:


# Transform different columns or subsets using ColumnTransformer
from sklearn.compose import ColumnTransformer

num_attrs = list(num_data)
cat_attrs = ["Origin"]

# complete pipeline to transform 
# both numerical and cat. attributes
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attrs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attrs),
    ])

prepared_data = full_pipeline.fit_transform(data)
prepared_data[0]


# # Part 2 ends here 

# In[618]:


# creating custom attribute adder class
acc_ix, hpower_ix, cyl_ix = 4,2, 0

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): # no *args or **kargs
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        
        return np.c_[X, acc_on_cyl]


# In[619]:


def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe 
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
        
    '''
    numerics = ['float64', 'int64']

    num_attrs = data.select_dtypes(include=numerics)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
        ])
    return num_attrs, num_pipeline


def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    nuerical and categorical data.
    
    Argument:
        data: original dataframe 
    Returns:
        prepared_data: transformed data, ready to use
    '''
    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
        ])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data


# In the above code snippet, we have cascaded a three transformations:
# 
# 1. Imputing Missing Values — using the SimpleImputer class discussed above.
# 2. Custom Attribute Addition— using the custom attribute class defined above.
# 3. Standard Scaling of each Attribute. Note: It is good practice to scale the values before feeding them to the ML model, using the `standardScaler` class.
# 

# Now, we will calling the final pipeline_transformer function defined above.

# In[620]:


# from raw data to processed data in 2 steps

preprocessed_df = preprocess_origin_cols(data)
prepared_data = pipeline_transformer(preprocessed_df)
prepared_data


# In[621]:


prepared_data[0]


# Now our data is ready to use to train our model! Here, we will use Linear Regression to make our initial prediction. In short, the flow is as follows:
# 1. Instantiate the model
# 2. Train the model using fit() method
# 3. Make predictions by passing the data through the pipeline transformer
# 4. Evaluate the model using Root Mean Squared Error (RMSE; standard for evaluating Regression models)

# In[666]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression() # instantiate the model class
lin_reg.fit(prepared_data, data_labels)

# testing the predictions with the first 5 rows
sample_data = data.iloc[:7]
sample_labels = data_labels.iloc[:7]

sample_data_prepared = pipeline_transformer(sample_data)

print("Prediction of samples: ", lin_reg.predict(sample_data_prepared))


# In[667]:


print("Actual Labels of samples: ", list(sample_labels))


# In[668]:


from sklearn.metrics import mean_squared_error

mpg_predictions = lin_reg.predict(prepared_data)
lin_mse = mean_squared_error(data_labels, mpg_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# Now, let's train a few more algorithms to compare their RMSE's:
# 1. Decision Tree
# 2. RandomForest
# 3. Support Vector Machine
# 

# In[669]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(prepared_data, data_labels)


# In[670]:


mpg_predictions = tree_reg.predict(prepared_data)
tree_mse = mean_squared_error(data_labels, mpg_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# Notice, we achieved an RMSE of 0, which is impossible because no ML model is perfect. The issue here is that we are testing our model on the same data we trained it on. Therefore, the model is overfitting the data. But we cannot use our test data we set aside earlier until we finalize our best performing model to go into production.

# To circumvent this, we will employ sklearn's cross-validation feature to randomly split the training set into K distinct subsets called folds. It then trains and evaluates the model K times, picking a different fold for evaluation every time and training on the other K-1 folds.
# 
# The result is an array containing the K evaluation scores.

# In[671]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, 
                         prepared_data, 
                         data_labels, 
                         scoring="neg_mean_squared_error", 
                         cv = 10)
tree_reg_rmse_scores = np.sqrt(-scores)

# all decision tree scores
tree_reg_rmse_scores


# In[672]:


tree_reg_rmse_scores.mean()


# In[673]:


scores = cross_val_score(lin_reg, prepared_data, data_labels, scoring="neg_mean_squared_error", cv = 10)
lin_reg_rmse_scores = np.sqrt(-scores)
lin_reg_rmse_scores


# In[674]:


lin_reg_rmse_scores.mean()


# Now, lets try 2 more models: a Random Forest and Support Vector Regressor respectively.

# In[675]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(prepared_data, data_labels)
forest_reg_cv_scores = cross_val_score(forest_reg,
                                         prepared_data,
                                         data_labels,
                                         scoring='neg_mean_squared_error',
                                         cv = 10)

forest_reg_rmse_scores = np.sqrt(-forest_reg_cv_scores)
forest_reg_rmse_scores.mean()


# In[676]:


from sklearn.svm import SVR

svm_reg = SVR(kernel='linear')
svm_reg.fit(prepared_data, data_labels)
svm_cv_scores = cross_val_score(svm_reg, prepared_data, data_labels,
                                scoring='neg_mean_squared_error',
                                cv = 10)
svm_rmse_scores = np.sqrt(-svm_cv_scores)
svm_rmse_scores.mean()


# Our Random Forest model has the lowest RMSE value, and, as a result, performs the best. However, it still needs to be fine-tuned. To do this efficiently, we will use GridSearchCV to find out the best combination of hyperparameters for our RandomForest model.

# In[677]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,
                           cv=10,
                          )

grid_search.fit(prepared_data, data_labels)


# In[678]:


grid_search.best_params_


# `GridSearchCV` requires you to pass the parameter grid. This is a python dictionary with parameter names as keys mapped with the list of values you want to test for that parameter.
# 
# So we can pass the model, scoring method, and cross-validation folds to it.
# 
# We can train the model and it will return the best parameters and results for each combination of parameters, below.

# In[679]:


cv_scores = grid_search.cv_results_

# printing all the parameters along with their scores
for mean_score, params in zip(cv_scores['mean_test_score'], cv_scores["params"]):
    print(np.sqrt(-mean_score), params)


# It is good practice to do a feature importance assessment to determine which features are most impactful for making predictions. We can accomplish this by enlisting the features and zipping them up with the `best_estimator`’s feature importance attribute as follows:

# In[680]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[681]:


extra_attrs = ["acc_on_power", "acc_on_cyl"]
numerics = ['float64', 'int64']
num_attrs = list(data.select_dtypes(include=numerics))

attrs = num_attrs + extra_attrs
sorted(zip(attrs, feature_importances), reverse=True)


# Interestingly, we see that acc_on_power, which is a feature we derived, has turned out to be the most important one.

# Finally, our model is ready to go. Now, we want to evaluate the entire system on our test data we side aside earlier.

# In[682]:


from sklearn import *

##capturing the best configuration
final_model = grid_search.best_estimator_

##segregating the target variable from test set
X_test = strat_test_set.drop("MPG", axis=1)
y_test = strat_test_set["MPG"].copy()

##preprocessing the test data origin column
X_test_preprocessed = preprocess_origin_cols(X_test)

##preparing the data with final transformation
X_test_prepared = pipeline_transformer(X_test_preprocessed)

##making final predictions
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse


# For convenience's sake, we can now create a function to cover the entire flow.

# In[465]:


def predict_mpg(config, model):
    
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    preproc_df = preprocess_origin_cols(df)
    prepared_df = pipeline_transformer(preproc_df)
    y_pred = model.predict(prepared_df)
    return y_pred


# And check our model's ability to predict on a random sample of vehicle configurations.

# In[ ]:


##checking it on a random sample
vehicle_config = {
    'Cylinders': [4, 6, 8],
    'Displacement': [155.0, 160.0, 165.5],
    'Horsepower': [93.0, 130.0, 98.0],
    'Weight': [2500.0, 3150.0, 2600.0],
    'Acceleration': [15.0, 14.0, 16.0],
    'Model Year': [81, 80, 78],
    'Origin': [3, 2, 1]
}

predict_mpg(vehicle_config, final_model)


# We are finally done training our model, now we must save it into a file for deployment. For that we will use the `pickle` library, which is perhaps the most common way to serializing objects in Python.

# In[466]:


import pickle

##saving the model
with open("model.bin", 'wb') as f_out:
    pickle.dump(final_model, f_out)
    f_out.close()


# In[467]:


##loading the model from the saved file
with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

predict_mpg(vehicle_config, model)


# To effectively deploy a model, we will need a file containing the trained model (which we have), a web service, and a cloud service provider. As part of our next steps (not detailed in this Notebook) we will use Flask to develop our web service and ultimately deploy our model to Heroku for simplicity's sake.








