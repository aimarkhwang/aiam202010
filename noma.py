import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt

model = tf.global_variables_initializer();


data = read_csv('./corona.csv', sep=',')

xy = np.array(data, dtype=np.float32)


x = xy[:, 1:-1]
y = xy[:, [-1]]
X = tf.placeholder(tf.float32, shape=None)
Y = tf.placeholder(tf.float32, shape=None)

W = tf.Variable(tf.random_normal([1]), 'weight')
b = tf.Variable(tf.random_normal([1]), 'bias')

hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00005)
train = optimizer.minimize(cost)




sess = tf.Session()

sess.run(tf.global_variables_initializer())



for step in range(10001):
    
    cost_, hypo_, tr_ = sess.run([cost, hypothesis, train], feed_dict={X: x, Y: y})

    if step % 100 == 0:

        print("epoch :", step, ", cost :", cost_)
        print("예측치: ", hypo_[0])
        costhistory.append(cost_)
        
saver = tf.compat.v1.train.Saver()

save_path = saver.save(sess, "./test/saved.cpkt")



