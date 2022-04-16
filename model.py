import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
##some configuration so that we can view everything in the console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

##importing the datasets

training_dataset = pd.read_csv("./train.csv")
test_dataset = pd.read_csv("./test.csv")
data = pd.concat([training_dataset.drop('SalePrice',axis=1),test_dataset],keys=['train','test'])
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

##next up we check for missing values
print(data.isnull().sum())

##and lets display a heatmap of missing values with seaborn
sns.heatmap(data.isnull(),yticklabels=False,cbar=False)
plt.show()

##with this data we can see that PoolQC, MiscFeature and Alley are almost always NA
##now lets look at the datatypes of the columns
print(data.info())

##now we can decide how to handle this data
##here we will fill lotfrontage, which has 486 missing values in both datasets with the mean of the lotfrontage, since it has float64 as datatype
data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())

##for the columns that have smaller amounts of missing data that are not quantitative, we can pick the most frequent value (the mode) for the columns and fill them with that
data['BsmtQual'] = data['BsmtQual'].fillna(data['BsmtQual'].mode()[0])
data['BsmtCond'] = data['BsmtCond'].fillna(data['BsmtCond'].mode()[0])
data['GarageType'] = data['GarageType'].fillna(data['GarageType'].mode()[0])
data['GarageFinish'] = data['GarageFinish'].fillna(data['GarageFinish'].mode()[0])
data['GarageQual'] = data['GarageQual'].fillna(data['GarageQual'].mode()[0])
data['GarageCond'] = data['GarageCond'].fillna(data['GarageCond'].mode()[0])

##next up we drop the alley, poolQC, fence, garageyrblt and miscfeature columns since these columns have a LOT of missing values
data.drop(['Alley','PoolQC','Fence','MiscFeature','GarageYrBlt', 'FireplaceQu'],axis=1,inplace=True)

print("New null data info after some cleaning up:")
print(data.isnull().sum())

##after printing again, we can see a few columns still left with some missing data
##these columns are MasVnrType, MasVnrArea, BsmtExposure, BsmtFinType1 and BsmtFinType2
##we will pick the mean again for the area, and the mode for the other columns
data['MasVnrType'] = data['MasVnrType'].fillna(data['MasVnrType'].mode()[0])
data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean())
data['BsmtExposure'] = data['BsmtExposure'].fillna(data['BsmtExposure'].mode()[0])
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(data['BsmtFinType1'].mode()[0])
data['BsmtFinType2'] = data['BsmtFinType2'].fillna(data['BsmtFinType2'].mode()[0])

##and now we can drop the remaining columns with 1 or 2 missing values, since this won't have such a big impact
data.dropna(inplace=True)

print("Final columns null values:")
print(data.isnull().sum())
sns.heatmap(data.isnull(),yticklabels=False,cbar=False)
plt.show()