#Testing with all features
# In version will Extract the relevant features to see the Impact


#--imports--#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


#--importing dataset--#

#importing dataset
ds_train_path = "..\\dataset\\train.csv"
ds_train = pd.read_csv(ds_train_path)
print(ds_train.columns) #printing column names

ds_test_path = "..\\dataset\\test.csv"
ds_test = pd.read_csv(ds_test_path)
print(ds_test.columns)

ds_train_test = pd.concat([ds_train,ds_test],sort ='false')
ds_train_test = ds_train_test[ds_train.columns] #unsorting column names



#--Extracting Features--#

####

#---splitting numerical and categorical data---#

#all_cols = x_train.columns # getting all column names #np.ndarray don't have .columns func so we have to use whole dataset
#num_cols = x_train._get_numeric_data().columns #getting column nmes of numeric column
all_cols = ds_train_test.columns # getting all column names
num_cols = ds_train_test._get_numeric_data().columns #getting column nmes of numeric column

#getting categorical col names
#using set as with list was getting error as we cannot subtract 2 list bt we can subtract two set###
set_all_cols = set(all_cols)
set_num_cols = set(num_cols)
set_categorical_cols = set_all_cols - set_num_cols

#converting column names to index numbers

#categorical
categorical_cols_index =[] #initializing empty list
for x in set_categorical_cols:    
        categorical_cols_index.append(ds_train_test.columns.get_loc(x)) #appending to list

#numeric
numeric_cols_index =[] #initializing empty list
temp_index =-1
for x in set_num_cols: 
    print(x)
    temp_index = ds_train_test.columns.get_loc(x)
    if temp_index != 80 : # have put this condition because as the y i.e sales price is also numerical and we were getting that columns indexx also but when imputer ws performed we were getting out of bound issue
        numeric_cols_index.append(temp_index)#appending to list
###



#fixing categorical data filling missing data
ds_train_test[list(set_categorical_cols)] = ds_train_test[list(set_categorical_cols)].fillna(ds_train_test.mode().iloc[0])



x_train_test = ds_train_test.iloc[:,:-1].values
y_train_test = ds_train_test.iloc[:,-1:].values



#---filling missing data---#
#-numeric-#
from sklearn.preprocessing import Imputer
imputer_obj = Imputer(missing_values = 'NaN', strategy ='mean' ,axis =0)
for x in numeric_cols_index:
    print(x)
    imputer_obj = imputer_obj.fit(x_train_test[:,x:x+1]) #error cannot convert string to float now no error
    x_train_test[:,x:x+1] = imputer_obj.transform(x_train_test[:,x:x+1])



#-------

#label encoder
#converting categorical to numeric using labelencoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_obj_x = LabelEncoder()

#labelencoder doesnot takes 2d array it only takes single column at at ime so
for x in categorical_cols_index:
    print("column is ",x)
    x_train_test[:,x] = labelencoder_obj_x.fit_transform(x_train_test[:,x]) 

#onehotencoder
    onehotencoder_x = OneHotEncoder(categorical_features=categorical_cols_index)
    x_train_test = onehotencoder_x.fit_transform(x_train_test).toarray()

#----------

x_train = x_train_test[:1460, :]
y_train = y_train_test[:1460, :]
x_test = x_train_test[1460: ,:]


    
from sklearn.linear_model import LinearRegression
multi_lin_reg = LinearRegression()
multi_lin_reg.fit(x_train,y_train)
y_train_pred = multi_lin_reg.predict(x_train)


y_test_pred = multi_lin_reg.predict(x_test)



#-----concatinating column for submissions-------#
ds_sub = pd.DataFrame([ds_test.iloc[:,0].values,y_test_pred[:,0]]).T
#-------writining excel usiing pandas-------#
from pandas import ExcelWriter
from pandas import ExcelFile


writer = ExcelWriter('prediction.xlsx')
ds_sub.to_excel(writer,'Sheet1',index=False)
writer.save()


