# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import dataset
dataSet = pd.read_csv('Salary_Data.csv')

print(dataSet.shape)
df = pd.DataFrame(dataSet)
# x and y axis values
x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:,-1].values
print("==========================")
print("x values = ", x)
print("==========================")
print("Y values = ", y)


# split dataset into training and test data

# here I am taking only 20% of whole dataset as test dataset
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2,random_state=0)

# Now I am going to train simple linear regression model on training data

Regression = LinearRegression()
Regression.fit(X_train,Y_train)

# now I want to see the prediction test results

Y_pred = Regression.predict(X_test)

# now we need to visualize the training set
print("graph")
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, Regression.predict(X_train), color = 'blue')
plt.title('Salaries vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# now we need to visualize the test set
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, Regression.predict(X_train), color = 'blue')
plt.title('Salaries vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()