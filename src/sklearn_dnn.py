import tensorflow as tf
import time
import numpy as np
import pandas as pd

#TO FIX? 

#loading data
print("loading data")
data_train = pd.read_csv("../data/numerai_training_data.csv")
data_pred = pd.read_csv("../data/numerai_tournament_data.csv")
features = ["feature"+str(i+1) for i in range(50)]
X_train = data_train[features]
y_train = data_train.target
X_pred = data_pred[features]


# execution
cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf_tflearn = tf.contrib.learn.DNNRegressor(hidden_units=[50,40,20],feature_columns=cols,model_dir="../model/skcompat/")
dnn = tf.contrib.learn.SKCompat(dnn_clf_tflearn)
dnn.fit(X_train, y_train,steps=5000)
print("Predicting...")
# Your trained model is now used to make predictions on the numerai_tournament_data
results = dnn.predict(X_pred)
results_df = pd.DataFrame(data={'probability':results['scores']})
joined = pd.DataFrame(data_pred.id).join(results_df)

print("Writing predictions to predictions.csv")
# Save the predictions out to a CSV file
joined.to_csv('predictions.csv', index=False)

