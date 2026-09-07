import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# create a smaple dataset
data={"semisters":[1,2,3,4,5,6,7,8],
      "percentage":[60,65,67,70,72,73,75,79]
      }

df=pd.DataFrame(data)

print("Dataset")
print(df)
