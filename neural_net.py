import numpy as np
import pickle

import sys
import argparse

import process_data
from linreg import Linear_Regression

import tensorflow as tf
seed=12
tf.set_random_seed(seed)



def main(_):

    training_epochs=1000
    ckpt_freq=200000 # define inverse check pointing frequency

    n_samples=10000
    train_size=8000
    validation_size=5000
    batch_size=200
    
    # ADAM learning params
    learning_rate=0.001 # learning rate
    beta1=0.9
    beta2=0.9999
    epsilon=1e-09

    opt_params=dict(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon)
    #opt_params=dict(learning_rate=learning_rate)
    param_str='/lr=%0.4f' %(learning_rate)

    # Import data
    protocols=process_data.read_data_sets(data_params,train_size=train_size,validation_size=validation_size)

    # define model
    model=Linear_Regression(max_t_steps,batch_size,opt_params)

    
    saver = tf.train.Saver() # defaults to saving all variables
    with tf.Session() as sess:

        """
        # restore most recent session from checkpoint directory
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) 
        """

        # Step 7: initialize the necessary variables, in this case, w and b
        sess.run(tf.global_variables_initializer())

        average_loss = 0.0
        # write summary
        #writer = tf.summary.FileWriter('./ising_reg'+param_str, sess.graph)

        # Step 8: train the model
        for index in range(training_epochs): 

            batch_X, batch_Y = protocols.train.next_batch(batch_size,seed=seed)

            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op],
                                                feed_dict={model.X: batch_X,model.Y: batch_Y} )
            # count training step
            step = sess.run(model.global_step)
            
            # add summary data to writer
            #writer.add_summary(summary, global_step=step)

            average_loss += loss_batch
            if (index + 1) % ckpt_freq == 0:
                saver.save(sess, './checkpoints/ising_reg', global_step=step)

            print(sess.run( model.loss, feed_dict={model.X: batch_X,model.Y: batch_Y}))

        # Step 9: test model
        print(sess.run(model.loss, feed_dict={model.X: protocols.test.data_X, model.Y: protocols.test.data_Y}) )
        print(sess.run(  [model.Y,model.Y_predicted] , feed_dict={model.X: protocols.test.data_X[9236:9238], model.Y: protocols.test.data_Y[9236:9238]}))
       


    
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

