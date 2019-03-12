import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from util import load_data_table, save_pointcloud, shuffle_data
import time
import math
from random import shuffle
from pyntcloud import PyntCloud
import trimesh
import pandas as pd
from plyfile import PlyData, PlyElement
import random

class PCBIGAN(object):

	def __init__(self,is_training,epoch,checkpoint_dir, learning_rate,z_dim,batch_size,beta1,beta2,pointcloud_dim):
		self.beta1 = beta1
		self.beta2 = beta2
		self.learning_rate = learning_rate
		self.z_dim = z_dim
		self.pointcloud_dim = pointcloud_dim
		self.training = is_training
		self.batch_size = batch_size
		self.epoch = epoch
		self.checkpoint_dir = checkpoint_dir
		self.table_fake = "table_fake"
		self.lam_gp = 10
		self.build_network()


		with tf.variable_scope("adam",reuse=tf.AUTO_REUSE) as scope:
			print("init_d_optim")
			self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.g_loss,var_list = self.vars_G_E)
			print("init_g_optim")
			self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.d_loss,var_list = self.vars_D)

		self.init  = tf.global_variables_initializer()
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True



	def encoder(self,y):
		with tf.variable_scope("encoder") as scope:
			y = tf.layers.conv1d(y,filters = 64,kernel_size=1,strides=1,padding="same",activation=tf.nn.leaky_relu,name = "enc_1")
			y = tf.layers.batch_normalization(y,momentum=0.99,epsilon=0.001)
			y = tf.layers.conv1d(y,filters = 128,kernel_size=1,strides=1,padding="same",activation=tf.nn.leaky_relu,name = "enc_2")
			y = tf.layers.batch_normalization(y,momentum=0.99,epsilon=0.001)
			y = tf.layers.conv1d(y,filters = 256,kernel_size=1,strides=1,padding="same",activation=tf.nn.leaky_relu, name = "enc_3")
			y = tf.layers.batch_normalization(y,momentum=0.99,epsilon=0.001)
			y = tf.layers.conv1d(y,filters = 512,kernel_size=1,strides=1,padding="same",activation=tf.nn.leaky_relu, name = "enc_4")
			y = tf.layers.batch_normalization(y,momentum=0.99,epsilon=0.001)
			y = tf.reduce_max(y,axis = 1)
			y = tf.layers.flatten(y)
			y = tf.layers.dense(y, 126,name = "enc_output")
		return y

	'''
	def generator(self,x):
		with tf.variable_scope("generator") as scope:
			x = tf.layers.dense(x, 126, activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name= "gen_1")
			x =tf.layers.batch_normalization(x,momentum=0.99,epsilon=0.001)
			x = tf.layers.dense(x, 256, activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name= "gen_2")
			x = tf.layers.batch_normalization(x,momentum=0.99,epsilon=0.001)
			x = tf.layers.dense(x, 512, activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name= "gen_3")
			x = tf.layers.batch_normalization(x,momentum=0.99,epsilon=0.001)
			x = tf.layers.dense(x, 1024, activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name= "gen_4")
			x = tf.layers.batch_normalization(x,momentum=0.99,epsilon=0.001)
			x = tf.layers.dense(x, 2048, activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name= "gen_5")
			x = tf.layers.batch_normalization(x,momentum=0.99,epsilon=0.001)
			x = tf.layers.dense(x, 3084, activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name= "gen_6")
			x = tf.reshape(x,[-1,1028,3])
		return x
		'''

	def generator(self,x):
		with tf.variable_scope("generator") as scope:
			x = tf.reshape(x,[-1,42,3])
			x = tf.contrib.nn.conv1d_transpose(x,filter=[1,84,3],output_shape=[-1,128,3],stride=1,padding="same",data_format= "NWC",name= "gen_1")
			x = tf.contrib.nn.conv1d_transpose(x,filter=[1,256,3],output_shape=[-1,256,3],stride=1,padding="same",data_format= "NWC",name= "gen_3")
			x = tf.contrib.nn.conv1d_transpose(x,filter=[1,512,3],output_shape=[-1,512,3],stride=1,padding="same",data_format= "NWC",name= "gen_4")
			x = tf.contrib.nn.conv1d_transpose(x,filter=[1,1028,3],output_shape=[-1,1028,3],stride=1,padding="same",data_format= "NWC",name= "gen_1")
		return x


	def discriminator(self,y,x,reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()
			x = tf.reshape(x,[-1,42,3])
			y = tf.concat([y,x],1)
			y = tf.layers.conv1d(y,filters = 64,kernel_size=1,strides=1,padding="same",activation=tf.nn.leaky_relu)
			y = tf.layers.conv1d(y,filters = 128,kernel_size=1,strides=1,padding="same",activation=tf.nn.leaky_relu)
			y = tf.layers.conv1d(y,filters = 64,kernel_size=1,strides=1,padding="same",activation=tf.nn.leaky_relu)
			y = tf.layers.flatten(y)
			y = tf.layers.dense(y, 1028,activation=tf.nn.leaky_relu, name="conv_2last")
			y = tf.layers.dense(y, 128,activation=tf.nn.leaky_relu, name="conv_1last")
			y = tf.layers.dense(y, 1, activation=tf.nn.sigmoid,name="conv_last")
		return y

	def build_network(self):
		eps = 1e-12
		self.input = tf.placeholder(tf.float32, [None,self.pointcloud_dim,3], name="real_pointcloud_data")
		self.z = tf.placeholder(tf.float32,[None,self.z_dim], name ="noice")
		self.Gen = self.generator(self.z)
		self.Enc = self.encoder(self.input)
		self.Dis_encoder = self.discriminator(self.input,self.Enc,reuse = False)
		self.Dis_generator = self.discriminator(self.Gen,self.z,reuse = True)

		#Tensorboard variables
		self.d_sum_real = tf.summary.histogram("d_real", self.Dis_encoder)
		self.d_sum_fake = tf.summary.histogram("d_fake", self.Dis_generator)
		self.Enc_sum = tf.summary.histogram("Enc", self.Enc)
		self.G_sum = tf.summary.histogram("G",self.Gen)
		self.z_sum = tf.summary.histogram("z_input",self.z)

		#Wassersteinmetrik
		#self.d_loss = tf.reduce_mean(self.Dis_encoder - self.Dis_generator)
		#self.g_loss = -tf.reduce_mean(self.Dis_generator - (1 - self.Dis_encoder))


		#Vanilla BI-GAN Loss
		self.d_loss = tf.reduce_mean(-tf.log(self.Dis_encoder) - tf.log(1.0 - self.Dis_generator))
		self.g_loss = tf.reduce_mean(-tf.log(self.Dis_generator) - tf.log(1.0 - self.Dis_encoder))

		tf.summary.scalar('self.g_loss', self.g_loss )
		tf.summary.scalar('self.d_loss', self.d_loss )

		#collect generator and encoder variables
		t_vars = tf.trainable_variables()
		self.vars_G_E = [var for var in t_vars if 'gen' in var.name or "enc" in var.name]
		self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')


		self.saver = tf.train.Saver()

		#Tensorboard variables
		self.summary_g_loss = tf.summary.scalar("g_loss",self.g_loss)
		self.summary_d_loss = tf.summary.scalar("d_loss",self.d_loss)

	def train(self):

		with tf.Session(config = self.config) as sess:
			#imported_meta = tf.train.import_meta_graph("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4.meta")
			#imported_meta.restore(sess, "C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4")
			train_writer = tf.summary.FileWriter("./logs",sess.graph)
			merged = tf.summary.merge_all()
			#test_writer = tf.summary.FileWriter("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/")
			self.counter = 1
			sess.run(self.init)
			self.training_data = load_data_table(self.pointcloud_dim,4)
			k = (len(self.training_data) // self.batch_size)
			self.start_time = time.time()
			loss_g_val,loss_d_val = 0, 0
			self.training_data = self.training_data[0:(self.batch_size*k)]
			test_counter = 0
			print("Lengh of the training_data:")
			print(len(self.training_data))
			for e in range(0,self.epoch):
				epoch_loss_d = 0.
				epoch_loss_g = 0.
				self.training_data = shuffle_data(self.training_data)
				for i in range(0,k):
					self.batch_z = np.random.uniform(0, 0.2, [self.batch_size, self.z_dim])
					self.batch = self.training_data[i*self.batch_size:(i+1)*self.batch_size]
					_, loss_d_val,loss_d = sess.run([self.d_optim,self.d_loss,self.summary_d_loss],feed_dict={self.input: self.batch, self.z: self.batch_z})
					train_writer.add_summary(loss_d,self.counter)
					_, loss_g_val,loss_g = sess.run([self.g_optim,self.g_loss,self.summary_g_loss],feed_dict={self.z: self.batch_z,self.input: self.batch})
					train_writer.add_summary(loss_g,self.counter)
					self.counter=self.counter + 1
					epoch_loss_d += loss_d_val
					epoch_loss_g += loss_g_val
				epoch_loss_d /= k
				epoch_loss_g /= k
				print("Loss of D: %f" % epoch_loss_d)
				print("Loss of G: %f" % epoch_loss_g)
				print("Epoch%d" %(e))

				if e % 10 == 0:
					#save_path = self.saver.save(sess,"C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt",global_step=e)
					#print("model saved: %s" %save_path)
					self.gen_noise = np.random.uniform(0, 0.2, [1, self.z_dim])
					code = sess.run([self.Enc], feed_dict={self.input: [self.batch[0]]})
					code = np.asarray(code)
					code = np.reshape(code,(1,self.z_dim))
					generator_output = sess.run([self.Gen], feed_dict={self.z: code})
					save_pointcloud(self.batch[0],test_counter,"input_encoder", self.pointcloud_dim)
					save_pointcloud(generator_output,test_counter,"encoder_encoder", self.pointcloud_dim)
					test_counter +=1
			print("training finished")
