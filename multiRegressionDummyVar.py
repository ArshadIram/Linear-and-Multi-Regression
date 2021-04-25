import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


dataSet = pd.read_csv('50_Startups.csv')
print(dataSet)
X = dataSet.iloc[:,:-1]
y = dataSet.iloc[:,4]

print("==============================================")
print("value of x", X)
print("==================dummy Variables======================")
# converting the
State = pd.get_dummies(X['State'], drop_first=True)
x=X.drop('State', axis=1)
X=pd.concat([x,State], axis=1)
print("Dummy Variables",X)

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
print("==================predicted and actual value======================")

print("predicted values", Y_pred)
print("==================Accuracy model======================")
score = r2_score(Y_test, Y_pred)
print(score*100)
# now concatenate two vectors of predicted and actual value vertically

