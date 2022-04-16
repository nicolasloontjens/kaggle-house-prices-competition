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


##next up, analysing the data and checking which values would be influencing the saleprice a lot
sns.set(rc={'figure.figsize':(12,8)})
sns.heatmap(data.corr(),vmin=0.5,square=True, linewidths=0.05, linecolor='white')
plt.show()
## with this heatmap, you can see that SalePrice is heavily influenced by:
#OverallQual
#GrLivArea
#
#and a bit by:
#TotalBsmtSF
#1stFlrSF
#and garagecars and area

##so with this we looked up how to make a more detailed heatmap and created this one:
top15cols = data.corr().nlargest(15,'SalePrice')['SalePrice'].index
top15corr = np.corrcoef(data[top15cols].values.T)
sns.heatmap(top15corr,yticklabels=top15cols.values,xticklabels=top15cols.values, annot=True, cbar=False)
plt.show()
##and here we can see the top 15 columns ranked by correlation

##first up we will look at the OverallQual
sns.boxplot(x=data['OverallQual'],y=data['SalePrice'])
plt.show()

##and this confirms that the overall quality definitely increases prices of the house
##next up, the GrLivArea:

plt.scatter(data['GrLivArea'],data['SalePrice'])
plt.xlabel("Above ground living area sqfeet")
plt.ylabel("Pricing")
plt.show()

##looking at this graph, we can see that it has a linear relationship and affects the SalePrice quite a bit
##now we will look at the basement sqfeet if there could also be a linear relationship

plt.scatter(data['TotalBsmtSF'],data['SalePrice'])
plt.xlabel("Basement area sqfeet")
plt.ylabel("Pricing")
plt.show()

##and on this graph we can see houses without a basement, and also houses with a basement
##we can see a linear relationship here, but it is way steeper than the living area 

##add garagecars/saleprice and garagearea/saleprice




##next up, we will split the features:
# we want to split the features containing sqfeet, prices, etc
# and the features with data containing "Excelllent, good, etc."
# we will then convert the 2nd list of features to a numerical scale since this will be better for our model
print(data.columns.tolist())

numericalfeatures = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
             'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'ExterQual', 'ExterCond', 
             'BsmtQual', 'BsmtCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC', '1stFlrSF', 
             '2ndFlrSF', 'LowQualFinSF','GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
             'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
             'Fireplaces', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
             'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'PoolArea', 'MiscVal',
             'YrSold']

gradingfeatures = ['OverallQual','OverallCond','GarageCond','GarageQual','ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual']  

textgrades = ['Ex','Gd','TA','Fa','Po']
numericalgrading = [9,7,5,3,1]
gradingdictionary = dict(zip(textgrades,numericalgrading))

data[gradingfeatures] = data[gradingfeatures].replace(gradingdictionary)

##next up we will select our categorical features, which are the opposite of the numerical ones, so we can turn them into dummy variables
categoricalfeatures = data.drop(numericalfeatures, axis=1).columns


dummies = pd.get_dummies(data[categoricalfeatures])
data.drop(categoricalfeatures,axis=1,inplace=True)
data = data.join(dummies)



##and now we will fit the data to the model, starting with LinearRegression
