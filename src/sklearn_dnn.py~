import tensorflow as tf
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#TO FIX? 

# cleaning
tf.reset_default_graph()

# construction : todo count np inputs based on nb of features columns
n_inputs = 50
n_hidden1 = 50
n_hidden2 = 40
n_hidden3 = 20
n_outputs = 2

X = tf.placeholder(tf.float32, shape=[None, 50], name="X")
y = tf.placeholder(tf.int64, shape=[None,2], name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("dnn"):
    # Dense Layer
    hidden1 = tf.layers.dense(inputs=X, units=n_hidden1, activation=tf.nn.relu)
    hidden2 = tf.layers.dense(inputs=hidden1, units=n_hidden2, activation=tf.nn.relu)
    hidden3 = tf.layers.dense(inputs=hidden2, units=n_hidden3, activation=tf.nn.relu)
    # Logits Layer
    logits = tf.layers.dense(inputs=hidden3, units=n_outputs)
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }


with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy, name="loss")
    
learning_rate = 0.001

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, tf.argmax(y,1), 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

#loading data
print("loading data")
all_data = pd.read_csv("../data/numerai_training_data.csv")
features = ["feature"+str(i+1) for i in range(50)]
eras = []
X_test = []
y_test = []
for era in all_data.era.unique():
    X_train, X_t, y_train, y_t = train_test_split(all_data[all_data.era == era][features], all_data[all_data.era == era].target, test_size=0.33, random_state=42)
    
    eras.append([X_train,pd.get_dummies(y_train)])
    X_test.append(X_t)
    y_test.append(pd.get_dummies(y_t))

X_test = pd.concat(X_test)
y_test = pd.concat(y_test)


# execution
n_epochs = 20
print("starting to learn")
timer = time.clock()
beginning = timer
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_train,y_train in eras:
            sess.run(training_op, feed_dict={X: X_train, y: y_train})
            acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        diff = time.clock()-timer
        timer = time.clock()
        print(epoch,": time elapsed","{0:.2f}".format(diff), "s train accuracy", acc_train, " est accuracy", acc_test)
        

    model_name = time.strftime("%Y-%M-%d_%H:%M")
    save_path = saver.save(sess, "../models/"+model_name+".ckpt")
end = time.clock()
total = end - beginning
print("Total duration",int(total//60),":",int(total%60))
