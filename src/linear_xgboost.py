import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

#todo era

print("Loading")
data_train = pd.read_csv("../data/numerai_training_data.csv")
data_pred = pd.read_csv("../data/numerai_tournament_data.csv")
features = ["feature"+str(i+1) for i in range(50)]
X = data_train[features]
y = data_train.target
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
X_pred = data_pred[features]
pipe = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85, objective='reg:linear',verbose=5)

#pipe = make_pipeline(PolynomialFeatures(2),XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.02, subsample=0.9, colsample_bytree=0.85, objective='reg:linear'))

print("Training")
model = pipe
model.fit(X, y)

print("Predicting...")
# Your trained model is now used to make predictions on the numerai_tournament_data
results = model.predict(X_pred)
results_df = pd.DataFrame(data={'probability':results})
joined = pd.DataFrame(data_pred.id).join(results_df)

print("Writing predictions to predictions.csv")
# Save the predictions out to a CSV file
joined.to_csv('predictions_linear_xgboost.csv', index=False)
