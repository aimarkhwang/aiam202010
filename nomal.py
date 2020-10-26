import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
model = tf.global_variables_initializer();

# 행은 non 열은 4
# 행은 non 열은 1
# 마지막값 (price)
# 4개의 변인을 입력을 받습니다.
#열의 마지막 값의 앞까지 인덱스 1부터 인덱스 1 = avg , 4개값
# 가격 값을 입력 받습니다.

# 플레이스 홀더를 설정합니다.
# with tf.name_scope("layer") as scope: 
#행렬곱 tf.matmul
data = read_csv('./onion_model/onion_price3.csv', sep=',')


xy = np.array(data, dtype=np.float32)
min_max_scaler = MinMaxScaler()
fitted = min_max_scaler.fit(xy)
std= StandardScaler()
fitted= std.fit(xy)
xy = std.transform(xy)
x = xy[:, 1:-1]
y = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")
hypothesis = tf.matmul(X, W) + b
    
w_hist = tf.summary.histogram("weight", W)
b_hist = tf.summary.histogram("bias", b)
hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# 가설을 설정합니다.


# 비용 함수를 설정합니다.
# with tf.name_scope("cost") as scope:

cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
cost_summ = tf.summary.scalar("cost", cost)



# 최적화 함수를 설정합니다.
# 0.000005
# with tf.name_scope("train") as scope:
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# 세션을 생성합니다.

sess = tf.Session()
    # tensorboard --logdir=./logs
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logt")
writer.add_graph(sess.graph)



# 글로벌 변수를 초기화합니다.

sess.run(tf.global_variables_initializer())

# predicted = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
# accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))

# 학습을 수행합니다.
costhistory = []
# hypohistory = []
# trhistory = []
whistory = []
for step in range(100001):
    summary, _ = sess.run([merged_summary, train], feed_dict={X: x, Y: y})
    writer.add_summary(summary, global_step=step)
    cost_, hypo_, tr_ = sess.run([cost, hypothesis, train], feed_dict={X: x, Y: y})

    if step % 100 == 0:

        print("epoch :", step, ", cost :", cost_)
        print("- 양파 가격: ", hypo_[0])
        # print("W : " , sess.run([W] ,feed_dict={X: x_data, Y: y_data}))
        costhistory.append(cost_)
        # hypohistory.append(hypo_)
        # trhistory.append(tr_)
        # whistory.append(sess.run([W] ,feed_dict={X: x_data, Y: y_data}))
# 학습된 모델을 저장합니다.
# cost1 = tf.reduce_mean(Y / hypothesis)
saver = tf.compat.v1.train.Saver()

save_path = saver.save(sess, "./test1/saved.cpkt")