import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split


#Load dataset
df=pd.read_csv("boston_housing.csv")


#View dimension of dataset
df.shape

#View some statistics of dataset mean,max,min etc
df.describe()

#View dataset features and attribute information
df.head()

# Assign feature variable to df_x 
df_x=df
print(df_x)


# Assign medv variable to df_y
df_y=df.price
print(df_y)


#Initialize the linear regression model
reg =linear_model.LinearRegression()



#Split the data into 67% training and 33% testing data
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)


#Train our model with the training data
reg.fit(x_train, y_train)

#Print the coefecients/weights for each feature/column of our model
print(reg.coef_)


#print our price predictions on our test data
y_pred = reg.predict(x_test)
print(y_pred)


#print the predicted price and actual price of houses from the testing data set row 0
y_pred[0]

y_test[0]

# To check the modeles performance / accuracy
print(np.mean((y_pred-y_test)**2))
