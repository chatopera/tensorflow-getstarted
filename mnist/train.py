# encoding: UTF-8
# copyright Hai Liang Wang
# RUN against tensorflow 0.11.0rc2

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(0)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=0)

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

XX = tf.reshape(X, [-1, 784])

Y = tf.nn.softmax(tf.matmul(XX, W) + b)
cross_entropy = -tf.reduce_mean(Y_* tf.log(Y)) * 100 * 10

correct_prediction = tf.equal(tf.argmax(Y,1 ), tf.argmax(Y_, 1))
accurancy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

def main():
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    batch_X, batch_Y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={X: batch_X, Y_:batch_Y})

if __name__ == '__main__':
    main()