import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataSet = pd.read_csv('50_Startups.csv')
print(dataSet)
x = dataSet.iloc[:,:-1].values
y = dataSet.iloc[:,-1].values

print("==============================================")
print("value of x", x)
print("==================One HotEncoding======================")
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X= np.array(ct.fit_transform(x))
print(X)

# we will use this new dataset X value as the categorical column is encoded into it
print("==================Spliting the Dataset======================")

X_train, X_test, Y_train, Y_test = train_test_split(X,y , test_size=0.2, random_state=0)

print("==================Model fit======================")
# build multiple linear regression
multiRegression = LinearRegression()
# train on training set
multiRegression.fit(X_train, Y_train)

print("==================Performance on new observations======================")
# This time we have several features so this time we need to visualization it as 2 vectors. 10 actual profits and 10 predicted profit
# predictions based on features

Y_pred = multiRegression.predict(X_test)
np.set_printoptions(precision = 2)
# now concatenate two vectors of predicted and actual value vertically
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)), 1))