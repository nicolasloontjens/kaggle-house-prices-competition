import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
##some configuration so that we can view everything in the console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

##importing the datasets

training_dataset = pd.read_csv("./train.csv")
test_dataset = pd.read_csv("./test.csv")
data = training_dataset
#pd.concat([training_dataset.drop('SalePrice',axis=1),test_dataset],keys=['train','test'])
data.drop('Id',axis=1, inplace=True)

"""
first we will check the data and make sure it is correct:
we should check the years, make sure that there aren't any sales that exceed the current year because that would be impossible
we should check the prices and measurements to see if they are not negative
"""

columns_with_years = ['YearBuilt','GarageYrBlt','YrSold','YearRemodAdd']
columns_with_measurements = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF', 
                             'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

##here we print the maximum years for a specific row and check if it exceeds 2022 => in this case we saw that the test dataset had a max of 2207 for GarageYrBlt
print(data[columns_with_years].max()) 

##here we check with an if statement if any columns with years in them exceed 2022, if so we grab the row
wrong_years = (data[columns_with_years] > 2022).any(axis=1)
print("rows with a wrong year:") 
print(data[wrong_years])

##in this situation we can see that the yearbuilt of the wrong garageyrblt entry is right so we replace garageyrblt with the yearblt value of the house
data.loc[wrong_years,'GarageYrBlt'] = data[wrong_years]['YearBuilt'] 

##now the years are all good 
print(data[columns_with_years].max())

##next up we will check the measurements
wrong_measurements = (data[columns_with_measurements] < 0).any(axis=1)

##this is empty, thus there are no wrong measurements
print("rows with impossible measurements:")
print(data[wrong_measurements])

##lets see what our data looks like again
print(data.shape)

##next up we will split the features and modify the data
# we want to split the features containing sqfeet, prices, etc
# and the features with data containing "Excelllent, good, etc."
# we will then convert the 2nd list of features to a numerical scale since this will be easier to use
numerical_features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
             'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
             'HeatingQC', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
             'GrLivArea', 'PoolArea', 'PoolQC', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'FireplaceQu', 'GarageYrBlt',
             'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'MiscVal','YrSold']

grading_features = ['OverallQual','OverallCond','GarageCond','GarageQual','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','PoolQC','FireplaceQu']  

text_grades = ['Ex','Gd','TA','Fa','Po']
numerical_grading = [9,7,5,3,1]
gradingdictionary = dict(zip(['Ex','Gd','TA','Fa','Po'],[9,7,5,3,1]))
##here we replace the Ex, Gd with 9, 7, etc., now we don't have to use dummies for these columns
data[grading_features] = data[grading_features].replace(gradingdictionary)

##these are the features that contain categories of data
categorical_features = data.drop(numerical_features,axis=1).columns

print(categorical_features)
##and now we will fit the data to the model, starting with LinearRegression
