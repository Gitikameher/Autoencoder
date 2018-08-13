from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data #Using mnist data for training and testing
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters for training
learning_rate = 0.01
num_steps = 30000
batch_size = 256
display_step = 1000
examples = 10

num_hidden_layer_1 = 256 # number of features in the first hidden layer
num_hidden_layer_2 = 128 # Second hidden layer features
num_input = 784 # data input (img shape: 28*28)

A = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_hl1': tf.Variable(tf.random_normal([num_input, num_hidden_layer_1])),
    'encoder_hl2': tf.Variable(tf.random_normal([num_hidden_layer_1, num_hidden_layer_2])),
    'decoder_hl1': tf.Variable(tf.random_normal([num_hidden_layer_2, num_hidden_layer_1])),
    'decoder_hl2': tf.Variable(tf.random_normal([num_hidden_layer_1, num_input])),
}
biases = {
    'encoder_bl1': tf.Variable(tf.random_normal([num_hidden_layer_1])),
    'encoder_bl2': tf.Variable(tf.random_normal([num_hidden_layer_2])),
    'decoder_bl1': tf.Variable(tf.random_normal([num_hidden_layer_1])),
    'decoder_bl2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder with sigmoid activation
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_hl1']), biases['encoder_bl1']))
    
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_hl2']), biases['encoder_bl2']))
    return layer_2


# Building the decoder with sigmoid activation
def decoder(x):
    
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_hl1']), biases['decoder_bl1']))
    
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_hl2']), biases['decoder_bl2']))
    return layer_2

# model construction
encoder_op = encoder(A)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
y_true = A

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optmzr = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

# Start Training with a tf session:

with tf.Session() as sess:
    sess.run(init)

    for i in range(1, num_steps+1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        _, l = sess.run([optmzr, cost], feed_dict={A: batch_x})
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    n = 4
    fig_orig = np.empty((28 * n, 28 * n))
    fig_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        batch_x, _ = mnist.test.next_batch(n)
        g = sess.run(decoder_op, feed_dict={A: batch_x})

        # Display original images
        for j in range(n):
            fig_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            fig_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(fig_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(fig_recon, origin="upper", cmap="gray")
    plt.show()
