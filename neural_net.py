import numpy as np
import pickle

import sys
import argparse

import process_data

import tensorflow as tf
seed=12
tf.set_random_seed(seed)



def main(_):
    # Import data
    protocols=process_data.read_data_sets(data_params)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 100])
    W = tf.Variable(tf.zeros([100,1]))
    b = tf.Variable(tf.zeros([1]))
    y = tf.matmul(x, W) + b #tf.minimum(tf.maximum( , -1) ,1)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1])


    #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    cross_entropy = tf.reduce_mean( tf.nn.l2_loss(y-y_))
    #train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.8, beta2=0.9999, epsilon=1e-08).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()


    # Train
    for _ in range(1000):
        batch_xs, batch_ys = protocols.train.next_batch(200,seed=seed)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print(sess.run( cross_entropy  , feed_dict={x: batch_xs, y_: batch_ys}))
    #exit()

    # Test trained model
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(  [y,y_] , feed_dict={x: protocols.test.data_X[9236:9238], y_: protocols.test.data_Y[9236:9238]}))

    
if __name__ == '__main__':

    # define number of samples 
    N_samples=100000
    max_t_steps = 100 # total number of time steps per protocol

    L=2 #spin chain system size
    dt=0.01 # time step
    T=1.0 # total ramp time

    # data dict
    data_params=dict(L=L,dt=dt,NT=int(T/dt))

    # run ML tool
    tf.app.run(main=main, argv=[sys.argv[0]] )

