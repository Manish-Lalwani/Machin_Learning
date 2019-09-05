#Multiple linear regression


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


dataset_path = "50_Startups.csv"
dataset = pd.read_csv(dataset_path)
x = dataset.iloc[:,:4].values #index starts from 0 last column or row number is excluded
#x = dataset.iloc[:,:-1]take all except last column
y = dataset.iloc[:,4:].values

#encoding the categorical data  state here (dummy variable)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#for changing categorical values to number 1,0
labelencoder_objx = LabelEncoder()
x[:,3]=labelencoder_objx.fit_transform(x[:,3])

#for dummy variable separate columns
onehotencoder_objx = OneHotEncoder(categorical_features=[3])
x = onehotencoder_objx.fit_transform(x).toarray()


#avoiding dummy variable trap by eliminating one caegorical column
x=x[:,1:]


#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)

#feature scalling
#no need to apply feature scalling will be taken care by the model


#fitting multiple linear regression model to the data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_test_predicted = regressor.predict(x_test)

compare = (y_test,y_test_predicted)



