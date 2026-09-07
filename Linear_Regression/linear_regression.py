import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# create a sample dataset
data={"semisters":[1,2,3,4,5,6,7],
      "percentage":[60,65,67,70,72,73,75]
      }

df=pd.DataFrame(data)

print("Dataset")
print(df)

# seperate features (X) and target(y)
X=df[["semisters"]]
y=df["percentage"]

# Spliting the dataset into training and testing
x_train,x_test,y_tain,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print("\nTraining Data:")
print(x_train)


print("\nTesting Data:")
print(x_test)

