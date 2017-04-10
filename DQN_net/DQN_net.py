import pickle

import sys
import argparse

import process_data
from linreg import Linear_Regression

import tensorflow as tf
seed=12
tf.set_random_seed(seed)

import numpy as np
import matplotlib.pyplot as plt



def main(_):

    training_epochs=30001
    ckpt_freq=200000 # define inverse check pointing frequency

    n_samples=100000
    train_size=80000
    validation_size=0
    batch_size=200
    
    # ADAM learning params
    learning_rate=0.004 # learning rate
    beta1=0.94
    beta2=0.88
    epsilon=1e-08

    opt_params=dict(learning_rate=learning_rate,beta1=beta1,beta2=beta2,epsilon=epsilon)
    #opt_params=dict(learning_rate=learning_rate)
    param_str='/lr=%0.4f' %(learning_rate)

    # Import data
    protocols=process_data.read_data_sets(data_params,train_size=train_size,validation_size=validation_size)

    # define model # 400, 60 -> 0.0001 in 3E4 epochs
    n_hidden_1=400
    n_hidden_2=60
    #n_hidden_3=20
    #model=Linear_Regression(max_t_steps,batch_size,opt_params)
    model=Linear_Regression(max_t_steps,batch_size,opt_params,n_hidden=(n_hidden_1,n_hidden_2))

    
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
        #writer = tf.summary.FileWriter('./fid_reg'+param_str, sess.graph)

        # Step 8: train the model
        for index in range(training_epochs): 

            batch_X, batch_Y = protocols.train.next_batch(batch_size,seed=seed)

            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op],
                                                feed_dict={model.X: batch_X,model.Y: batch_Y   } )
            # count training step
            step = sess.run(model.global_step)

            
            # add summary data to writer
            #writer.add_summary(summary, global_step=step)

            average_loss += loss_batch/batch_size
            if (index + 1) % ckpt_freq == 0:
                saver.save(sess, './checkpoints/fid_reg', global_step=step)

            print(index,loss_batch/batch_size)

            if (index+1)%training_epochs==0 and index!=0:
        
                # Step 9: test model
                train_loss, train_Y, train_Y_predicted = sess.run([model.loss/train_size, model.Y, model.Y_predicted], feed_dict={model.X: protocols.train.data_X, model.Y: protocols.train.data_Y})
                print("train loss:", train_loss)

                test_loss, test_Y, test_Y_predicted = sess.run([model.loss/(n_samples-train_size), model.Y, model.Y_predicted], feed_dict={model.X: protocols.test.data_X, model.Y: protocols.test.data_Y})
                print("test loss:", test_loss)

                Y = np.concatenate((train_Y, test_Y))
                Y_predicted = np.concatenate((train_Y_predicted, test_Y_predicted))

                
                plt.scatter(train_Y,train_Y_predicted,color='b')
                plt.scatter(test_Y,test_Y_predicted,color='r')
                plt.show()

        
        print(sess.run( model.Y, feed_dict={model.X: protocols.test.data_X[9236:9238], model.Y: protocols.test.data_Y[9236:9238]}))
        print(sess.run( model.Y_predicted, feed_dict={model.X: protocols.test.data_X[9236:9238], model.Y: protocols.test.data_Y[9236:9238]}))
        print(sess.run( [tf.reduce_min( tf.abs(model.Y-model.Y_predicted) ),tf.reduce_max( tf.abs(model.Y-model.Y_predicted))], feed_dict={model.X: protocols.test.data_X, model.Y: protocols.test.data_Y}))
              
        

    
if __name__ == '__main__':

    # define number of samples 
    N_samples=100000
    max_t_steps = 100 # total number of time steps per protocol

    L=2 #spin chain system size
    dt=0.05 # time step
    T=1.5 # total ramp time

    # data dict
    data_params=dict(L=L,dt=dt,NT=int(T/dt))

    # run ML tool
    tf.app.run(main=main, argv=[sys.argv[0]] )

