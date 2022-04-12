# pip install numpy
# pip install pandas
# pip install matplotlib


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('./train.csv')

#print(data.head())

#print(data.describe())

##scatterplot showing price/sqfeet
plt.scatter(data.LotArea,data.SalePrice,s=5)
plt.title("Price vs Square Feet")
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.show()


##boxplot showing pricing/neighborhood
n_price = {}

for i in data['Neighborhood']:
    n_price[i] = data[data.Neighborhood == i].SalePrice

plt.boxplot([x for x in n_price.values()],labels=[x for x in n_price.keys()]) 
plt.show()

##we should add a barchart showing nr of sales/neighborhood as well