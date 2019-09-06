#Testing with all features
# In version will Extract the relevant features to see the Impact

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns


#importing dataset and splitting

#call x_train,y_train = import_split_dataset_func()
def import_split_dataset_func():
    # importing data set
    ds_train_path = "..\\dataset\\train.csv"
    ds_train = pd.read_csv(ds_train_path)
    print(ds_train.columns)

    #splitting dataset into x_train and y_train
    x_train = ds_train.iloc[:,:-1].values
    y_train = ds_train.iloc[:,-1].values

    return x_train,y_train



#choosing features
#splitting numerical and categorical data

#call  set_categorical_cols = split_num_categorical_cols_func(x_train)
def split_num_categorical_cols_func(x_train):   
    #all_cols = x_train.columns # getting all column names
    #num_cols = x_train._get_numeric_data().columns #getting column nmes of numeric column
    all_cols = ds_train.columns # getting all column names
    num_cols = ds_train._get_numeric_data().columns #getting column nmes of numeric column

    ###using set as with list was getting error as we cannot subtract 2 list bt we can subtract two set###
    set_all_cols = set(all_cols)
    set_num_cols = set(num_cols)
    set_categorical_cols = set_all_cols - set_num_cols

    #converting column names to index numbers
    	#categorical
    categorical_cols_index =[] #initializing empty list
    for x in set_categorical_cols:    
        categorical_cols_index.append(ds_train.columns.get_loc(x)) #appending to list
    
    	#numeric
    #converting column_names to column numbers or indices
	numeric_cols_index =[] #initializing empty list
	temp_index =-1
	for x in set_num_cols: 
    	print(x)
    	temp_index = ds_train.columns.get_loc(x)
    	if temp_index != 80 : # have put this condition because as the y i.e sales price is also numerical and we were getting that columns indexx also but when imputer ws performed we were getting out of bound issue
            numeric_cols_index.append(temp_index)#appending to list

    return categorical_cols_index,numeric_cols_index




#filling missing data
	#numeric
from sklearn.preprocessing import Imputer
imputer_obj = Imputer(missing_values = 'NaN', strategy ='mean' ,axis =0)

for x in numeric_cols_index:
    print(x)
    imputer_obj = imputer_obj.fit(x_train[:,x:x+1]) #error cannot convert string to float
    x_train[:,x:x+1] = imputer_obj.transform(x_train[:,x:x+1])

    #categorical



#fixing categorical data
ds_train[list(set_categorical_cols)] = ds_train[list(set_categorical_cols)].fillna(ds_train.mode().iloc[0])


#fixing space in column 3rd column MsZoning
###temp_store = x_train[:,2]
###x_train[:,2] = np.char.strip(x_train[:,2]," ")
###np.core.defchararray.strip(x_train[:,2], chars=None)
###np.char.replace(x_train[:,2]," ","")
ds_train.iloc[:,2] = ds_train.iloc[:,2].str.replace(" ","") #worked


#label encoder
#converting categorical to numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_obj_x = LabelEncoder()

#labelencoder doesnot takes 2d array it only takes single column at at ime so
for x in categorical_cols_index:
    print("column is ",x)
    x_train[:,x] = labelencoder_obj_x.fit_transform(x_train[:,x]) 

#onehotencoder
    onehotencoder_x = OneHotEncoder(categorical_features=categorical_cols_index)
    x_train = onehotencoder_x.fit_transform(x_train).toarray()
#trainig model and predicting
def multi_lin_reg_train_func(x_train,y_train):
    from sklearn.linear_model import LinearRegression
    multi_lin_reg = LinearRegression()
    multi_lin_reg.fit(x_train,y_train)
    y_pred = multi_lin_reg.predict(x_train)
    return y_pred


#evaluating model
#ref https://medium.com/acing-ai/how-to-evaluate-regression-models-d183b4f5853d
#call evaluate_func(y_train,y_pred)
def evaluate_func(y_train,y_pred):
    #using Mean Absolute Error (MAE)
    from sklearn.metrics import mean_absolute_error
    print("Mean absolute error is :" ,mean_absolute_error(y_train,y_pred))
