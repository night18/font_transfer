'''
=======================================================================================
Author: Chun-Wei Chiang
Date: 2019.04.24
Description: Font transfer using CycleGAN
=======================================================================================
Change logs
=======================================================================================
'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Conv2D,Conv2DTranspose, Activation, concatenate, MaxPool2D, Flatten, Dense, BatchNormalization, ZeroPadding2D, Dropout, LeakyReLU, GlobalAveragePooling2D
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Lambda
from data_loader import DataLoader
import datetime
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

gen_filter_num = 32
dis_filter_num = 64
batch_size = 1
sample_interval = 100
epochs = 50
pool_size = 50
img_width = 48
img_height = 48
img_depth = 1
relu_alpha = 0.2
dropout_rate = 0.2
g_learning_rate = 0.005
d_learning_rate = 0.001
lambda_cycle = 10
img_shape = (img_height, img_width, img_depth)

class CycleGAN():
	def __init__(self):


		self.dataset_name = 'font2font'
		self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res = img_shape)

		# Calculate output shape of D (PatchGAN)
		patch = 3
		self.disc_patch = (patch, patch, 1)

		g_optimizer = Adam(g_learning_rate)
		d_optimizer = Adam(d_learning_rate)
		
		# Build and compile the discriminator
		self.d_A = self.discriminator(name='d_A')
		self.d_B = self.discriminator(name='d_B')
		self.d_A.compile(
			loss = 'mse',
			optimizer = d_optimizer,
			metrics = ['accuracy']
			)
		self.d_B.compile(
			loss = 'mse',
			optimizer = d_optimizer,
			metrics = ['accuracy']
			)

		# Build the generator
		self.g_AtoB = self.generator(name='g_AtoB')
		self.g_BtoA = self.generator(name='g_BtoA')

		# Inputs
		img_A = Input(shape=img_shape)
		img_B = Input(shape=img_shape)

		# Change domain
		fake_B = self.g_AtoB(img_A)
		fake_A = self.g_BtoA(img_B)

		# Reconstruct to original domain
		recon_A = self.g_BtoA(fake_B)
		recon_B = self.g_AtoB(fake_A)

		# stop training discriminator when training generator
		d_A_layers = [l for l in self.d_A.layers]

		for i in range(len(d_A_layers)):
			d_A_layers[i].trainable = False

		d_B_layers = [l for l in self.d_B.layers]
		for i in range(len(d_B_layers)):
			d_B_layers[i].trainable = False
		# self.d_A.trainable = False
		# self.d_B.trainable = False

		# validate whether the generator can create true like image
		valid_A = self.d_A(fake_A)
		valid_B = self.d_B(fake_B)
		
		self.combine = Model(
						inputs = [img_A, img_B],
						outputs = [valid_A, valid_B, recon_A, recon_B]
						)

		self.combine.compile(
						loss = ['mse', 'mse', 'mae', 'mae'],
						loss_weights = [1, 1, lambda_cycle, lambda_cycle],
						optimizer = g_optimizer
						)
		
	def generator(self, name = ''):
		def shortcutModule(pre_lyr, output_channels, res_id, is_stride = False):
			if is_stride:
				shortcut = MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same', name=res_id+'_shortcut_pool')(pre_lyr)
			else:
				shortcut = pre_lyr

			# shortcut = Conv2D(output_channels, kernel_size=(1,1), padding='same', name=res_id+'shortcut_conv')(shortcut)
			return shortcut

		def myResModule(pre_lyr, kernel_length, first_channels, second_channels, res_id, is_stride =False):
			if is_stride:
				x = Conv2D(first_channels, kernel_size=(kernel_length,kernel_length), strides=(2,2) ,padding='same', name=res_id+'_conv_1' )(pre_lyr)
			else:
				x = Conv2D(first_channels, kernel_size=(kernel_length,kernel_length), strides=(1,1) ,padding='same', name=res_id+'_conv_1' )(pre_lyr)
			x = BatchNormalization(name=res_id+'_batch_1')(x)
			x = Dropout(dropout_rate, name=res_id+'_dropout_1')(x)
			x = LeakyReLU(alpha=relu_alpha)(x)

			x = Conv2D(second_channels, kernel_size=(kernel_length,kernel_length), strides=(1,1), padding='same', name=res_id+'_conv_2')(x)
			
			shortcut = shortcutModule(pre_lyr, second_channels, res_id, is_stride)
			x = concatenate([x, shortcut], name=res_id+'_conc')
			x = BatchNormalization(name=res_id+'_batch_2')(x)
			x = Dropout(dropout_rate, name=res_id+'_dropout_2')(x)
			x = LeakyReLU(alpha=relu_alpha)(x)
			return x

		def encoder(pre_lyr, name=''):
			x = Conv2D(gen_filter_num, kernel_size=(7,7), strides=(1,1), padding='same', name= name+'_encoder_conv_1')(pre_lyr)
			x = BatchNormalization(name= name+'_encoder_norm_1')(x)
			x = LeakyReLU(alpha=relu_alpha, name= name+'_encoder_relu_1')(x)

			x = Conv2D( 2 * gen_filter_num , kernel_size=(3,3), strides=(2,2), padding='same', name= name+'_encoder_conv_2')(x)
			x = BatchNormalization(name= name+'_encoder_norm_2')(x)
			x = LeakyReLU(alpha=relu_alpha, name= name+'_encoder_relu_2')(x)

			x = Conv2D( 4 * gen_filter_num, kernel_size=(3,3), strides=(2,2), padding='same', name= name+'_encoder_conv_3')(x)
			x = BatchNormalization(name= name+'_encoder_norm_3')(x)
			x = LeakyReLU(alpha=relu_alpha, name= name+'_encoder_relu_3')(x)
			
			return x


		def transfer(pre_lyr, name=''):
			x = myResModule(pre_lyr, 3, 4 * gen_filter_num, 4 * gen_filter_num, name+'_res_1')
			x = myResModule(x, 3, 4 * gen_filter_num, 4 * gen_filter_num, name+'_res_2')
			x = myResModule(x, 3, 4 * gen_filter_num, 4 * gen_filter_num, name+'_res_3')
			x = myResModule(x, 3, 4 * gen_filter_num, 4 * gen_filter_num, name+'_res_4')
			x = myResModule(x, 3, 4 * gen_filter_num, 4 * gen_filter_num, name+'_res_5')
			x = myResModule(x, 3, 4 * gen_filter_num, 4 * gen_filter_num, name+'_res_6')

			return x

		def decoder(pre_lyr, name=''):
			x = Conv2DTranspose( 2 * gen_filter_num, kernel_size=(3,3), strides=(2,2), padding='same', name=name+'_decoder_conv_1')(pre_lyr)
			x = BatchNormalization(name= name+'_decoder_norm_1')(x)
			x = LeakyReLU(alpha=relu_alpha, name= name+'_decoder_relu_1')(x)

			x = Conv2DTranspose( gen_filter_num, kernel_size=(3,3), strides=(2,2), padding='same', name=name+'_decoder_conv_2')(x)
			x = BatchNormalization(name= name+'_decoder_norm_2')(x)
			x = LeakyReLU(alpha=relu_alpha, name= name+'_decoder_relu_2')(x)

			x = Conv2DTranspose( img_depth, kernel_size=(7,7), activation='tanh' ,strides=(1,1), padding='same', name=name+'_decoder_conv_3')(x)
			return x

		img = Input(shape=img_shape)

		x1 = encoder(img, name=name)
		x2 = transfer(x1, name=name)
		output_gen = decoder(x2, name=name)

		model = Model(
			inputs = img,
			outputs = output_gen)

		return model

	def discriminator(self, name = ''):

		def d_layer(pre_lyr, filters, kernel_size=(3,3), name=''):
			x = Conv2D(filters, kernel_size=kernel_size, strides=(4,4), padding='same', name= 'disc_conv_'+ name)(pre_lyr)
			x = BatchNormalization(name= 'disc_norm_'+ name)(x)
			x = LeakyReLU(alpha=relu_alpha, name= 'disc_relu_'+ name)(x)
			return x

		img = Input(shape=img_shape)

		x1 = d_layer(img, dis_filter_num, kernel_size=(5,5), name='1')
		x2 = d_layer(x1, 2 * dis_filter_num, kernel_size=(5,5), name='2')
		# The discriminator is too strong
		# x3 = d_layer(x2, 4 * dis_filter_num, kernel_size=(3,3), name='3')
		# x4 = d_layer(x3, 4 * dis_filter_num, kernel_size=(3,3), name='4')

		prediction = Conv2D( 1, kernel_size=(3,3), name= name+'_disc_conv_pred')(x2)
		model = Model(
			inputs = img,
			outputs = prediction
		)
		return model


	def train(self, epochs, batch_size = 1, sample_interval = 200):
		# TODO make the parameter self

		start_time = datetime.datetime.now()

		# Adversarial loss ground truths
		# shape = (1, 3, 3, 1)
		valid = np.ones((batch_size, ) + self.disc_patch )
		fake = np.zeros((batch_size, ) + self.disc_patch )

		for epoch in range(epochs):
			acc_sum = 0
			for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

				# Train Discriminator

				fake_B = self.g_AtoB.predict(imgs_A)
				fake_A = self.g_BtoA.predict(imgs_B)

				dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
				dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
				dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

				dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
				dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
				dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

				d_loss = 0.5 * np.add(dA_loss, dB_loss)

				# self.d_A.summary()


				g_loss = self.combine.train_on_batch([imgs_A, imgs_B],
													[valid, valid,
													imgs_A, imgs_B])
				# self.d_A.summary()

				elapsed_time = datetime.datetime.now() - start_time

				# plot the progress
				'''
				print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f] time: %s " \
														% ( epoch, epochs,
															batch_i, self.data_loader.n_batches,
															d_loss[0], 100*d_loss[1],
															g_loss[0],
															np.mean(g_loss[1:3]),
															np.mean(g_loss[3:5]),
															elapsed_time))
															'''
				acc_sum += 100 * d_loss[1]
				# If at save interval => save generated image samples
				if batch_i % (self.data_loader.n_batches - 2) == 0:
					print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f] time: %s " \
														% ( epoch, epochs,
															batch_i, self.data_loader.n_batches,
															d_loss[0], 100*d_loss[1],
															g_loss[0],
															np.mean(g_loss[1:3]),
															np.mean(g_loss[3:5]),
															elapsed_time))
					self.sample_images(epoch, batch_i)
			print( "mean accuracy of epoch %d is %3d%%" %(epoch, acc_sum/self.data_loader.n_batches ))

		self.store_model('models')

	def sample_images(self, epoch, batch_i):
		os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
		r, c = 2, 3

		imgs_A = self.data_loader.load_data(domain='A', batch_size=1, is_testing=True)
		imgs_B = self.data_loader.load_data(domain='B', batch_size=1, is_testing=True)

		fake_B = self.g_AtoB.predict(imgs_A)
		fake_A = self.g_BtoA.predict(imgs_B)

		recon_A = self.g_BtoA.predict(fake_B)
		recon_B = self.g_AtoB.predict(fake_A)

		gen_imgs = np.concatenate([imgs_A, fake_B, recon_A, imgs_B, fake_A, recon_B])

		#print(gen_imgs)
		# rescale the images to 0 -1
		gen_imgs = 0.5 * gen_imgs + 0.5

		titles = ['Original', 'Translated', 'Reconstructed']
		fig, axs = plt.subplots(r, c)
		cnt = 0

		for i in range(r):
			for j in range(c):
				display = gen_imgs[cnt].reshape(gen_imgs[cnt].shape[0],gen_imgs[cnt].shape[1])
				axs[i,j].imshow(display, cmap='gray')
				axs[i,j].set_title(titles[j])
				axs[i,j].axis('off')
				cnt += 1
		fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
		plt.close()


	def store_model(slef, path):
		g_AtoB_h5_storage_path = path + '/ga2b.h5'
		g_BtoA_h5_storage_path = path + '/gb2a.h5'

		d_A_h5_storage_path = path + '/da.h5'
		d_B_h5_storage_path = path + '/db.h5'

		save_model(
			self.g_AtoB,
			g_AtoB_h5_storage_path,
			overwrite = True,
			include_optimizer=True
		)
		save_model(
			self.g_BtoA,
			g_BtoA_h5_storage_path,
			overwrite = True,
			include_optimizer=True
		)

		save_model(
			self.d_A,
			d_A_h5_storage_path,
			overwrite = True,
			include_optimizer=True
		)
		save_model(
			self.d_B,
			d_B_h5_storage_path,
			overwrite = True,
			include_optimizer=True
		)


	def load_model(self, path):
		g_AtoB_h5_storage_path = path + '/ga2b.h5'
		g_BtoA_h5_storage_path = path + '/gb2a.h5'

		d_A_h5_storage_path = path + '/da.h5'
		d_B_h5_storage_path = path + '/db.h5'
		error = False
		try:
			self.g_AtoB = load_model(
				g_AtoB_h5_storage_path,
				compile=True
			)

			self.g_BtoA = load_model(
				g_BtoA_h5_storage_path,
				compile=True
			)

			self.d_A = load_model(
				d_A_h5_storage_path,
				compile=True
			)

			self.d_B = load_model(
				d_B_h5_storage_path,
				compile=True
			)

		except Exception as e:
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			error = True
			print(e)
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		finally:
			return error

if __name__ == '__main__':

	config = ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.90
	with tf.Session(config=config) as sess:
		tf.set_random_seed(100)
		gan = CycleGAN()
		if gan.load_model('models'):
			gan.train(epochs=epochs, batch_size = 1,  sample_interval=sample_interval)
