import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from util import load_data_art,shuffle_data,save_image
import time
import math
from random import shuffle
import random

class CGAN(object):

	def __init__(self,is_training,epoch,checkpoint_dir,learning_rate,z_dim,batch_size,beta1,beta2,image_size):
		""""
		Args:
			beta1: beta1 for AdamOptimizer
			beta2: beta2 for AdamOptimizer
			learning_rate: learning_rate for the AdamOptimizer
			training: [bool] Training/NoTraining
			batch_size: size of the batch_
			epoch: number of epochs
			checkpoint_dir: directory in which the model will be saved
			name_art: name of the fake_art will be saved
			image_size: size of the image
			z_dim: sample size from the normal distribution for the generator
		"""

		self.beta1 = beta1
		self.beta2 = beta2
		self.learning_rate = learning_rate
		self.training = is_training
		self.batch_size = batch_size
		self.epoch = epoch
		self.checkpoint_dir = checkpoint_dir
		self.name_art = "fake_art"
		self.image_size = image_size
		self.z_dim = z_dim
		self.build_network()

	def generator(self,x):
		with tf.variable_scope("generator") as scope:

			x = tf.layers.dense(x,1024,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),name="gan_input_layer")
			x = tf.nn.leaky_relu(x)
			x = tf.layers.batch_normalization(x)
			x = tf.reshape(x, [-1, 1, 1, 1024])
			#size 5 output_size = strides * (input_size-1) + kernel_size - 2*padding padding = valid padding = 0
			x = tf.layers.conv2d_transpose(x,filters=512,kernel_size=5,strides=2,padding='valid',name="gan_deconv_1")
			x = tf.nn.leaky_relu(x)
			x = tf.layers.batch_normalization(x)
			#size 13
			x = tf.layers.conv2d_transpose(x,filters=256,kernel_size=5,strides=2,padding='valid',name="gan_deconv_2")
			x = tf.nn.leaky_relu(x)
			x = tf.layers.batch_normalization(x)
			#size 29
			x = tf.layers.conv2d_transpose(x,filters=128,kernel_size=5,strides=2,padding='valid',name="gan_deconv_3")
			x = tf.nn.leaky_relu(x)
			x = tf.layers.batch_normalization(x)
			#size 61
			x = tf.layers.conv2d_transpose(x,filters=64,kernel_size=5,strides=2,padding='valid',name="gan_deconv_4")
			x = tf.nn.leaky_relu(x)
			x = tf.layers.batch_normalization(x)
			#size 125
			x = tf.layers.conv2d_transpose(x,filters=3,kernel_size=8,strides=2,padding='valid',name="gan_deconv_5")
			x = tf.nn.tanh(x)
			#x = tf.reshape(x,[self.batch_size,self.image_size,self.image_size,3])
		return x


	def discriminator(self,x,reuse=False):
		with tf.variable_scope("discriminator" ,reuse=reuse):
			#x = tf.reshape(x,[self.batch_size,self.image_size,self.image_size,3])
			x = tf.layers.conv2d(x,filters=128,kernel_size=5,padding='valid',strides=(2,2),activation = tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),name="conv_1")
			#tf.layers.batch_normalization(x)
			x = tf.layers.conv2d(x,filters=128,kernel_size=5,activation = tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),strides=2,name="conv_2")
			tf.layers.batch_normalization(x)
			x = tf.layers.conv2d(x,filters=256,kernel_size=5,activation = tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),strides=2,name="conv_3")
			tf.layers.batch_normalization(x)
			x = tf.layers.conv2d(x,filters=512,kernel_size=5,activation = tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),strides=2,name="conv_4")
			tf.layers.batch_normalization(x)
			x = tf.layers.conv2d(x,filters=1024,kernel_size=5,activation = tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),strides=2,name="conv_5")
			tf.layers.batch_normalization(x)
			x = tf.layers.flatten(x)
			x = tf.layers.dense(x,1,activation=tf.nn.sigmoid,kernel_initializer=tf.contrib.layers.xavier_initializer(),name="disc_output")

		return x

	def build_network(self):

		self.input = tf.placeholder(tf.float32, [self.batch_size,self.image_size,self.image_size,3], name="real_art_picture")
		self.z = tf.placeholder(tf.float32,[None,self.z_dim], name ="noice")
		self.Gen = self.generator(self.z)
		self.Dis_real = self.discriminator(self.input,reuse = False)
		self.Dis_generator = self.discriminator(self.Gen,reuse = True)

		#Tensorboard variables
		self.d_sum_real = tf.summary.histogram("d_real", self.Dis_real)
		self.d_sum_fake = tf.summary.histogram("d_fake", self.Dis_generator)

		self.G_sum = tf.summary.histogram("G",self.Gen)
		self.z_sum = tf.summary.histogram("z_input",self.z)

		#Wassersteinmetrik
		#self.d_loss = tf.reduce_mean(self.Dis_encoder - self.Dis_generator)
		#self.g_loss = -tf.reduce_mean(self.Dis_generator - (1 - self.Dis_encoder))


		#Vanilla BI-GAN Loss
		self.d_loss = -tf.reduce_mean(tf.log(self.Dis_real) + tf.log(1. - self.Dis_generator))
		self.g_loss = -tf.reduce_mean(tf.log(self.Dis_generator))

		tf.summary.scalar('self.g_loss', self.g_loss )
		tf.summary.scalar('self.d_loss', self.d_loss )

		#collect generator and encoder variables
		t_vars = tf.trainable_variables()
		self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
		self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

		self.saver = tf.train.Saver()

		#Tensorboard variables
		self.summary_g_loss = tf.summary.scalar("g_loss",self.g_loss)
		self.summary_d_loss = tf.summary.scalar("d_loss",self.d_loss)

	def train(self):
		with tf.variable_scope("adam",reuse=tf.AUTO_REUSE) as scope:
			print("init_d_optim")
			self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.g_loss,var_list = self.vars_G)
			print("init_g_optim")
			self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.d_loss,var_list = self.vars_D)

			self.init  = tf.global_variables_initializer()
			self.config = tf.ConfigProto()
			self.config.gpu_options.allow_growth = True

		with tf.Session(config = self.config) as sess:
			#imported_meta = tf.train.import_meta_graph("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4.meta")
			#imported_meta.restore(sess, "C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4")
			train_writer = tf.summary.FileWriter("./logs",sess.graph)
			merged = tf.summary.merge_all()
			#test_writer = tf.summary.FileWriter("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/")
			self.counter = 1
			sess.run(self.init)
			self.training_data = load_data_art()
			print(self.training_data.shape)
			k = (len(self.training_data) // self.batch_size)
			self.start_time = time.time()
			loss_g_val,loss_d_val = 0, 0
			self.training_data = self.training_data[0:(self.batch_size*k)]
			test_counter = 0
			for e in range(0,self.epoch):
				epoch_loss_d = 0.
				epoch_loss_g = 0.
				self.training_data = shuffle_data(self.training_data)
				for i in range(0,k):
					self.batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])
					self.batch = self.training_data[i*self.batch_size:(i+1)*self.batch_size]
					self.batch = np.asarray(self.batch)
					_, loss_d_val,loss_d = sess.run([self.d_optim,self.d_loss,self.summary_d_loss],feed_dict={self.input: self.batch, self.z: self.batch_z})
					train_writer.add_summary(loss_d,self.counter)
					_, loss_g_val,loss_g = sess.run([self.g_optim,self.g_loss,self.summary_g_loss],feed_dict={self.z: self.batch_z, self.input: self.batch})
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
					self.gen_noise = np.random.uniform(-1, 1, [1, self.z_dim])
					fake_art = sess.run([self.Gen], feed_dict={self.z: self.gen_noise})
					save_image(fake_art,self.name_art,test_counter)

					test_counter +=1
			print("training finished")
