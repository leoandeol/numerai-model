from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data = pd.read_csv("../data/numerai_training_data.csv")
features = ["feature"+str(i+1) for i in range(50)]
X = data[features]
y = data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

model = LinearRegression()
model.fit(X_train,y_train)
print("R=",np.sqrt(model.score(X_test,y_test)))
