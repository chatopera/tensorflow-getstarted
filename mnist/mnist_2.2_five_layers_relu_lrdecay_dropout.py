# encoding: UTF-8
# Copyright 2016 Google.com
# https://github.com/martin-gorner/tensorflow-mnist-tutorial
#
# Modifications copyright (C) 2017 Hai Liang Wang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import tensorflow as tf
from tqdm import tqdm
from utils import gen_model_save_dir
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

# neural network with 5 layers
#
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 784]   # 784 = 28*28
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (sigmoid)      W1 [784, 200]      B1[200]
#  · · · · · · · · ·                                             Y1 [batch, 200]
#   \x/x\x/x\x/x\x/      -- fully connected layer (sigmoid)      W2 [200, 100]      B2[100]
#    · · · · · · ·                                               Y2 [batch, 100]
#    \x/x\x/x\x/         -- fully connected layer (sigmoid)      W3 [100, 60]       B3[60]
#     · · · · ·                                                  Y3 [batch, 60]
#     \x/x\x/            -- fully connected layer (sigmoid)      W4 [60, 30]        B4[30]
#      · · ·                                                     Y4 [batch, 30]
#      \x/               -- fully connected layer (softmax)      W5 [30, 10]        B5[10]
#       ·                                                        Y5 [batch, 10]

# Download images and labels into mnist.test (10K images+labels) and
# mnist.train (60K images+labels)
mnist = read_data_sets("MNIST_data", one_hot=True,
                       reshape=False, validation_size=0)
# input X: 28x28 grayscale images, the first dimension (None) will index
# the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no
# dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)

# five layers and their number of neurons (tha last layer has 10 softmax
# neurons)
L = 200
M = 100
N = 60
O = 30
# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive*
# values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.zeros([N]))
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.zeros([O]))
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# The model, with dropout at each layer
XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)

Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
Y3d = tf.nn.dropout(Y3, pkeep)

Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100
tf.scalar_summary('loss', cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary('accuracy', accuracy)

# training step, learning rate = 0.003
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
model_save_dir = gen_model_save_dir(prefix='2.2_five_layers_relu_lrdecay_dropout')
tf_writer = tf.train.SummaryWriter(model_save_dir)
tf_saver = tf.train.Saver(max_to_keep=200)  # Arbitrary limit
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

merged_summaries = tf.merge_all_summaries()
tf_writer.add_graph(sess.graph)


def run(i, update_test_data, update_train_data):
    '''
    You can call this function in a loop to train the model, 100 images at a time
    '''
    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0  # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    learning_rate = min_learning_rate + \
        (max_learning_rate - min_learning_rate) * math.exp(-i / decay_speed)

    # compute training values for visualisation
    if update_train_data:
        a, c, s = sess.run([accuracy, cross_entropy, merged_summaries], {
                           X: batch_X,
                           Y_: batch_Y,
                           pkeep: 1.0})
        tf_writer.add_summary(s, i)
        # a, accuracy
        # c, loss
        # s, summaries

    # compute test values for visualisation
    if update_test_data:
        a, c = sess.run([accuracy, cross_entropy], {
                        X: mnist.test.images,
                        Y_: mnist.test.labels,
                        pkeep: 1.0})
        # a, test accuracy
        # c, test loss

    # the backpropagation training step
    sess.run(train_step, {
        X: batch_X,
        Y_: batch_Y,
        lr: learning_rate,
        pkeep: 0.75})


def main():
    pbar = tqdm(range(2000))
    for k in pbar:
        epoch = str(k * 100 // mnist.train.images.shape[0] + 1)
        pbar.set_description('Processing epoch %s' % epoch)
        run(k + 1, True, True)

    print("Saving model to %s ..." % model_save_dir)
    tf_saver.save(sess, '%s/model.ckpt' % model_save_dir)
    print("Done.")

if __name__ == '__main__':
    main()
