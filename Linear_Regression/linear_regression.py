import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# create a sample dataset
data={"hours":[1,2,3,4,5,6,7,8,9,10],
      "drill_meters":[6,10,14,19,25,30,37,43,49,56]
      }

df=pd.DataFrame(data)

print("Dataset")
print(df)

# seperate features (X) and target(y)
X=df[["hours"]]
y=df["drill_meters"]

# Spliting the dataset into training and testing
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("\nTraining Data:")
print(x_train)


print("\nTesting Data:")
print(x_test)

# create the model
model=LinearRegression()

#train the model
model.fit(x_train,y_train)

#Make Predictions
y_pred=model.predict(x_test)

print("\nActual Values")
print(y_test.values)

print("\nPredicted Values")
print(y_pred)

#Evaluate the model
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("\nModel Evaluation")
print("Mean Absolute Error:",mae)
print("Mean Square Error:",mse)
print("Root Mean Square Error:",rmse)
print("r2 Score:",r2)