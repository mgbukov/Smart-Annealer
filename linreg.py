# -*- coding: utf-8 -*-
import sys, os

import tensorflow as tf
#seed=12
#tf.set_random_seed(seed)

import numpy as np


class Linear_Regression(object):
	# build the graph for the model
	def __init__(self,n_feats,n_samples,opt_params,n_hidden=()):

		# define global step for checkpointing
		self.global_step=tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		
		self.n_feats=n_feats
		self.n_samples=n_samples
		if len(n_hidden):
			self.hidden=True
			self.n_hidden_1=n_hidden[0]
			self.n_hidden_2=n_hidden[1]
		else:
			self.hidden=False

		# Step 1: create placeholders for input X and label Y
		self._create_placeholders()
		# Step 2: create weight and bias, initialized to 0 and construct model to predict Y from X
		self._create_model()
		# Step 3: define loss function
		self._create_loss()
		# Step 4: use gradient descent to minimize loss
		self._create_optimiser(opt_params)
		# Step 5: create sumamries
		self._create_summaries()

		self._measure_accuracy()


	def _create_placeholders(self):
		with tf.name_scope('data'):
			self.X=tf.placeholder(tf.float32, shape=(None,self.n_feats), name="X_data")
			self.Y=tf.placeholder(tf.float32, shape=(None,1), name="Y_data")

	def _create_model(self):
		with tf.name_scope('model'):
			if self.hidden:
				# hidden layer 1
				self.W1 = tf.Variable( tf.random_normal(shape=(self.n_feats,self.n_hidden_1), ),dtype=tf.float32, name="weight_1")
				self.b1 = tf.Variable( tf.random_normal(shape=[self.n_hidden_1]), name="bias_1")
				self.layer_1 = tf.nn.relu( tf.add( tf.matmul(self.X,self.W1), self.b1 ) )
				# hidden layer 2
				self.W2 = tf.Variable( tf.random_normal(shape=(self.n_hidden_1,self.n_hidden_2), ),dtype=tf.float32, name="weight_2")
				self.b2 = tf.Variable(tf.random_normal(shape=[self.n_hidden_2]), name="bias_2")
				self.layer_2 = tf.nn.relu( tf.add( tf.matmul(self.layer_1,self.W2), self.b2 ) )
				# output layer
				self.W = tf.Variable( tf.random_normal(shape=(self.n_hidden_2,1), ),dtype=tf.float32, name="weight_out")
				self.b = tf.Variable(tf.random_normal(shape=[1]), name="bias_out")
				# define model
				self.Y_predicted=tf.add( tf.matmul(self.layer_2,self.W), self.b )
			else:
				# output layer
				self.W = tf.Variable( tf.random_normal(shape=(self.n_feats,1), ),dtype=tf.float32, name="weight_out")
				self.b = tf.Variable(tf.random_normal(shape=[1]), name="bias_out")
				
				# define model
				self.Y_predicted=tf.add( tf.matmul(self.X,self.W), self.b )
			
			
	def _create_loss(self):
		with tf.name_scope('loss'):
			#self.loss = tf.reduce_sum(tf.pow(self.Y - self.Y_predicted, 2))/(2.0*self.n_samples)
			self.loss = tf.reduce_mean( tf.nn.l2_loss(self.Y - self.Y_predicted)) \
						+ 1.98*tf.reduce_mean( tf.abs(self.W) )
			
	def _create_optimiser(self,kwargs):
		with tf.name_scope('optimiser'):
			#self.optimizer = tf.train.GradientDescentOptimizer(**kwargs).minimize(self.loss,global_step=self.global_step)
			self.optimizer = tf.train.AdamOptimizer(**kwargs).minimize(self.loss,global_step=self.global_step)

	def _measure_accuracy(self):
		"""to be written"""
		with tf.name_scope('accuracy'):
			pass

	def _create_summaries(self):
		with tf.name_scope("summaries"):
			tf.summary.scalar("loss", self.loss)
			#tf.summary.scalar("accuracy", self.accuracy)
			tf.summary.histogram("histogram loss", self.loss)
			# merge all summaries into one op to make it easier to manage
			self.summary_op = tf.summary.merge_all()

