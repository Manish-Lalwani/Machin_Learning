
#Polynomila regression



#importing libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

dataset_path = "Position_Salaries.csv"
os.path.exists(dataset_path)

dataset = pd.read_csv(dataset_path)

#x = dataset.iloc[:,1].values #if you only specify thes then x will be vector(as we are only taking sing column but we want x to be matrix always so we will rewrite this as
x = dataset.iloc[:,1:2].values #same as but know it will take as matrix
y= dataset.iloc[:,2].values

#Splitting the data into Training and Test set
#here we have very small dataset only 10 rows so we will not split data as very less data is given
#rom sklearn.preprocessing import train_test_split
#_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state =0)



#Feature scalling
#no need as we are using multiple linear regression which automatically does it
#from sklearn.preprocessing import StandardScaler
#standard_scaler = StandardScaler()
#x_train = standard_scaler.fit_transform(x_train)
#x_test = standard_scaler.fit_transform(x_test)


#also fitting linear regression for comparison with polynomial linear regression
#Fitting Linear Regression model to the dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x,y)
y_predict_linear_regression = linear_regressor.predict(x)


#Fitting Polynomial Regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_features = PolynomialFeatures(degree =2)
x_poly = polynomial_features.fit_transform(x)

from sklearn.linear_model import LinearRegression
linear_regressor2 = LinearRegression()
linear_regressor2.fit(x_poly,y)
y_predict_polynomial_regression = linear_regressor2.predict(x_poly)



#Visualizing Linear Regression result
plt.scatter(x,y, color = 'red')
plt.plot(x,y_predict_linear_regression)
plt.title("Results using linear regression")
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()

#Visualizing Polynomial Regression result
plt.scatter(x,y ,color ='green')
plt.plot(x,y_predict_polynomial_regression)
plt.show()




#polynomial with degree 4

#Fitting Polynomial Regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_features = PolynomialFeatures(degree =4)
x_poly4 = polynomial_features.fit_transform(x)

from sklearn.linear_model import LinearRegression
linear_regressor4 = LinearRegression()
linear_regressor4.fit(x_poly4,y)
y_predict_polynomial_regression4 = linear_regressor4.predict(x_poly4)


#Visualizing Polynomial Regression result
plt.scatter(x,y ,color ='green')
plt.plot(x,y_predict_polynomial_regression4)
plt.show()

#as we can see from result canging polynomial degree from 2 to 4 changed the result tremenduosly



#advanced graph
#here the graph is fine but it's showing direct lines in between i.e. from 0-1 ,1-2,2-3...9-10 so to make it more smooth will divide x which is from 0-10 to 0,0.1,0.2....9.9,10 will decrese the interval gap
#for this will use numpy
#and new x will be named as x_grid
x_grid = np.arange(start = min(x),stop = max(x),step = 0.1) #returned vector
x_grid = x_grid.reshape(len(x_grid),1) # return matrices of 90 rpws one column

#plotting graph with x_grid
plt.scatter(x,y ,color ='green')
plt.plot(x_grid,linear_regressor4.predict(polynomial_features.fit_transform(x_grid)), color = 'blue')
plt.show()


#predicting a single result in this case
#in this casepredicting result of an employee having 6.5 years of experience
#will take the regression which is already trained and use the predict method


#Linear regression result
linear_regressor.predict([[6.5]])
#polynomial linear regression result
#here we used the polynomial regressor with polynomialfeature of degree 4
#so we cannot pass value directly here as we have to first add the degree using polynomial feature
#and then the regressor can calculate
temp =polynomial_features.fit_transform([[6.5]])
linear_regressor4.predict(temp)

