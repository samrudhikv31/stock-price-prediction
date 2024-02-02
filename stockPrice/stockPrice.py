#import libary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error,r2_score


#get data
df = pd.read_csv('TSLA.csv')
print(df)
#data information
df.head()
df.shape
df.describe()
df.info()
#tesla close price graph
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
df.head()

#implementing decision tree regressor algorithm
features = ['High', 'Open', 'Low']
target = 'Close'
X = df[features]
Y = df[target]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)

#graph of actual and predicted
plt.scatter(X_test['High'], Y_test, color='black', label='Actual')
plt.scatter(X_test['High'], Y_pred, color='blue', label='Predicted')
plt.xlabel('High')
plt.ylabel('Close')
plt.legend()
plt.title('Actual vs Predicted Close Prices')
plt.show()


#getting mean_square_value(MSE) and R-square value
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
tree.plot_tree(regressor.fit(X_train,Y_train))

#testing the machine by giving our own value for high,open and low
a=input("enter high value")
b=input("enter open value")

c=input("enter low value")


testing_case = pd.DataFrame({'High': [a], 'Open': [b], 'Low': [c]})

# Get predictions for the testing case
prediction = regressor.predict(testing_case)
#getting close value
print(f'predicted close value is: {prediction[0]:.2f}')

