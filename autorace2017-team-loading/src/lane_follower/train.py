#!/bin/python3
import tensorflow as tf 
import numpy as np
#print(np.loadtxt('TrainingData.txt', delimiter=',', dtype=np.float32))
#data = np.loadtxt('./TrainingData.txt', delimiter=',', dtype=np.float32)
#print(data[:3])
data = np.genfromtxt('./TrainingData.txt', delimiter=',')[:]
print(data)
data_len=int(data.shape[0]/784) * 784
data_in = np.reshape(data[:data_len],(-1,784)) # type low size of training data instead of 10831
#data_in = np.reshape(data[:,:-1],(10831,784)) # type low size of training data instead of 10831
data_out = data_in[:,-1:]

# parameters
learning_rate = 0.001
training_epochs = 1000

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 1])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
#sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())
#sess.run(tf.variables_initializer(global_variables())
saver = tf.train.Saver(keep_checkpoint_every_n_hours = 1)
# train my model
for epoch in range(training_epochs):
    feed_dict = {X: data_in, Y: data_out, keep_prob: 0.7}
    c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    avg_cost = c

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')
save_path = saver.save(sess, "model/model.ckpt")
