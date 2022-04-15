import numpy as np
import pandas as pd
##importing the datasets
training_dataset = pd.read_csv("./train.csv")
test_dataset = pd.read_csv("./test.csv")
data = pd.concat([training_dataset.drop('SalePrice',axis=1),test_dataset],keys=['train','test'])
data.drop('Id',axis=1, inplace=True)

"""
first we will check the data and make sure it is correct:
we should check the years, make sure that there aren't any sales that exceed the current year because that would be impossible
we should check the prices, areas and distances to see if they are not negative
and we should make sure that the months like MoSold are correct and not an impossible value like 13
"""

columns_with_years = ['YearBuilt','GarageYrBlt','YrSold','YearRemodAdd']
columns_with_measurements = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF', 
                             'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

##here we print the maximum years for a specific row and check if it exceeds 2022 => in this case we saw that the test dataset had a max of 2207 for GarageYrBlt
print(data[columns_with_years].max()) 

##here we check with an if statement if any columns with years in them exceed 2022, if so we grab the row
wrong_values = (data[columns_with_years] > 2022).any(axis=1)

print(data[wrong_values])

## in this situation we can see that the yearbuilt of the wrong garageyrblt entry is right so we replace garageyrblt with the yearblt value of the house 
data.loc[wrong_values,'GarageYrBlt'] = data[wrong_values]['YearBuilt'] 

## now the years are all good 
print(data[columns_with_years].max())

