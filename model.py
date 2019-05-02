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
from tensorflow.keras.layers import Input, Conv2D,Conv2DTranspose, GaussianNoise, Activation, concatenate, MaxPool2D, Flatten, Dense, BatchNormalization, ZeroPadding2D, Dropout, LeakyReLU, GlobalAveragePooling2D
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Lambda
from data_loader import DataLoader
import datetime
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

gen_filter_num = 16
dis_filter_num = 16
ecd_filter_num = 32
batch_size = 1
sample_interval = 1000
epochs = 50
pool_size = 50
img_width = 256
img_height = 256
img_depth = 1
relu_alpha = 0.3
dropout_rate = 0.2
g_learning_rate = 5e-5
s_learning_rate = 0.001
d_learning_rate = 5e-5
lambda_cycle = 20
lambda_id = 0.1 * lambda_cycle
lambda_style = 1
lambda_encode = 0.5
img_shape = (img_height, img_width, img_depth)
use_tanh = True
disc_slower = False
use_style = False
use_autoencoder = False
dg_ratio = 5

class CycleGAN():
	def __init__(self):


		self.dataset_name = 'sketch2photo'
		# self.dataset_name = 'apple2orange'
		# self.dataset_name = 'font2font'
		self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res = img_shape, use_tanh=use_tanh)

		# Calculate output shape of D (PatchGAN)
		patch = int(img_width / 2**4)
		self.disc_patch = (patch, patch, 1)

		self.g_optimizer = Adam(g_learning_rate, 0.05)
		self.d_optimizer = Adam(d_learning_rate, 0.05)
		
		if use_style:
			# Train Style siamese
			self.style_siamese = self.load_siamese_model()
			if self.style_siamese == None:
				self.style_siamese = self.style_checker()
				self.style_siamese.compile(
					loss = self.siameseLoss,
					optimizer = SGD(lr=0.001)
				)


				siamese_train = self.data_loader.create_style_siamese()
				siamese_train_base = [ i[0] for i in siamese_train]
				siamese_train_pair = [ i[1] for i in siamese_train]
				siamese_train_label = np.array([ i[2] for i in siamese_train])

				hist = self.style_siamese.fit(
					[siamese_train_base, siamese_train_pair],
					siamese_train_label,
					epochs = epochs,
					batch_size = 32,
					validation_split = 0.3,
					verbose= 1)

				del siamese_train
				del siamese_train_base
				del siamese_train_pair
				del siamese_train_label

				#save the model
				save_model(
					self.style_siamese,
					"models/siamese.h5",
					overwrite=True,
					include_optimizer=True
				)

		if use_autoencoder:
			self.auto_encoder = self.load_autoender_model()
			if self.auto_encoder == None:
				self.auto_encoder = self.autoEncoder()
				self.auto_encoder.compile(
					loss = ['mse','mse'],
					loss_weights = [1, 0],
					optimizer = SGD(lr = 0.001)
					)


				zeros = np.zeros((batch_size))


				auto_encoder_train = self.data_loader.load_all_img()
				hist = self.auto_encoder.fit(
					auto_encoder_train,
					[auto_encoder_train, auto_encoder_train],
					epochs = 30,
					batch_size = 32,
					verbose = 0)

				del auto_encoder_train

				save_model(
					self.auto_encoder,
					'models/autoencoder.h5',
					overwrite = True,
					include_optimizer = True)

		# Build and compile the discriminator
		self.d_A = self.discriminator(name='d_A')
		self.d_B = self.discriminator(name='d_B')
		self.d_A.compile(
			loss = 'mse',
			optimizer = self.d_optimizer,
			metrics = ['accuracy']
			)
		self.d_B.compile(
			loss = 'mse',
			optimizer = self.d_optimizer,
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

		# fake_B = GaussianNoise(0.3)(fake_B)
		# fake_A = GaussianNoise(0.3)(fake_A)


		# Reconstruct to original domain
		recon_A = self.g_BtoA(fake_B)
		recon_B = self.g_AtoB(fake_A)

		img_A_id = self.g_BtoA(img_A)
		img_B_id = self.g_AtoB(img_B)

		# stop training discriminator when training generator
		d_A_layers = [l for l in self.d_A.layers]

		for i in range(len(d_A_layers)):
			d_A_layers[i].trainable = False

		d_B_layers = [l for l in self.d_B.layers]
		for i in range(len(d_B_layers)):
			d_B_layers[i].trainable = False

		if use_style:
			s_layers = [l for l in self.style_siamese.layers]
			for i in range(len(s_layers)):
				s_layers[i].trainable = False

		if use_autoencoder:
			a_layers = [l for l in self.auto_encoder.layers]
			for i in range(len(a_layers)):
				a_layers[i].trainable = False
		# self.d_A.trainable = False
		# self.d_B.trainable = False

		# validate whether the generator can create true like image
		valid_A = self.d_A(fake_A)
		valid_B = self.d_B(fake_B)



		if use_style:
			style_A = self.style_siamese([img_A, fake_A])
			style_B = self.style_siamese([img_B, fake_B])
		
			self.combine = Model(
							inputs = [img_A, img_B],
							outputs = [valid_A, valid_B, recon_A, recon_B, img_A_id, img_B_id, style_A, style_B],
							name = 'combine'
							)

			# self.combine.summary()

			self.combine.compile(
							loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae', self.siameseLoss, self.siameseLoss],
							loss_weights = [1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id, lambda_style, lambda_style],
							optimizer = self.g_optimizer
							)
		elif use_autoencoder:
			encoding_fake_B = self.auto_encoder(fake_B)[1]
			encoding_fake_A = self.auto_encoder(fake_A)[1]

			self.combine = Model(
							inputs = [img_A, img_B],
							outputs = [valid_A, valid_B, recon_A, recon_B, img_A_id, img_B_id, encoding_fake_B, encoding_fake_A],
							name = 'combine'
							)
			self.combine.compile(
							loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae', 'mae', 'mae'],
							loss_weights = [1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id, lambda_encode, lambda_encode],
							optimizer = self.g_optimizer
							)

		else:
			self.combine = Model(
							inputs = [img_A, img_B],
							outputs = [valid_A, valid_B, recon_A, recon_B, img_A_id, img_B_id],
							name = 'combine'
							)

			# self.combine.summary()

			self.combine.compile(
							loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
							loss_weights = [1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id],
							optimizer = self.g_optimizer
							)
	
	def autoEncoder(self):
		img = Input(shape=img_shape)
		x = Conv2D(gen_filter_num, kernel_size=(3,3), strides=(1,1), padding='same', name= 'autoencoder_conv_1')(img)
		x = BatchNormalization(name= 'autoencoder_norm_1')(x)
		x = LeakyReLU(alpha=relu_alpha, name= 'autoencoder_encoder_relu_1')(x)

		x = Conv2D( 2 * gen_filter_num , kernel_size=(3,3), strides=(2,2), padding='same', name= 'autoencoder_conv_2')(x)
		x = BatchNormalization(name= 'autoencoder_encoder_norm_2')(x)
		x = LeakyReLU(alpha=relu_alpha, name= 'autoencoder_encoder_relu_2')(x)

		x = Conv2D( 4 * gen_filter_num, kernel_size=(3,3), strides=(2,2), padding='same', name= 'autoencoder_encoder_conv_3')(x)
		x = BatchNormalization(name= 'autoencoder__encoder_norm_3')(x)
		encode_img = LeakyReLU(alpha=relu_alpha, name= 'autoencoder__encoder_relu_3')(x)
		x = Conv2DTranspose( 2 * gen_filter_num, kernel_size=(4,4), strides=(2,2), padding='same', name='autoencoder__decoder_conv_1')(encode_img)
		x = BatchNormalization(name= 'autoencoder_decoder_norm_1')(x)
		x = LeakyReLU(alpha=relu_alpha, name='autoencoder__decoder_relu_1')(x)

		x = Conv2DTranspose( gen_filter_num, kernel_size=(4,4), strides=(2,2), padding='same', name='autoencoder_decoder_conv_2')(x)
		x = BatchNormalization(name= 'autoencoder_decoder_norm_2')(x)
		x = LeakyReLU(alpha=relu_alpha, name= 'autoencoder_decoder_relu_2')(x)

		output_gen = Conv2DTranspose( img_depth, kernel_size=(4,4), activation='tanh' ,strides=(1,1), padding='same', name='autoencoder_decoder_conv_3')(x)
		
		model = Model(
			inputs = img,
			outputs = [output_gen, encode_img],
			name='autoEncoder')

		return model


	def siameseLoss(self, yTrue, yPred):
		# yTrue is label, and yPred is distance
		return K.mean( (1-yTrue) * K.square(yPred) + yTrue * K.square(K.maximum(1 - yPred, 0)))

	def load_autoender_model(self):

		try:
			model = load_model(
				'models/autoencoder.h5',
				compile=True
			)

		except Exception as e:
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			model = None
			print(e)
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")

		finally:
			return model


	def load_siamese_model(self, learning_rate = 0.001):
		try:
			model = load_model(
				"models/siamese.h5",
				custom_objects={'siameseLoss':self.siameseLoss},
				compile=True
			)

		except Exception as e:
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			model = None
			print(e)
			print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")

		finally:
			return model

	def style_checker(self):
		def eucl_dist_output_shape(shapes):
			shape1, shape2 = shapes
			return (shapes[0], shapes[2])

		def fire_module(prv_lyr, fire_id, squeeze = 3, expand = 4):
			s_id = 'fire' + str(fire_id) + '/'
			sqz = 'sqz1'
			relu = 'relu_'
			exp1 = 'exp1'
			exp3 = 'exp3'

			#squeeze layer
			sqz_layer = Conv2D( squeeze, kernel_size=(1,1), padding='same', name=s_id+sqz )(prv_lyr)
			sqz_layer = Activation( 'relu', name=s_id+relu+sqz )(sqz_layer)

			#expand layer
			#1*1
			exp1_layer = Conv2D( expand, kernel_size=(1,1), padding='same', name=s_id+exp1)(sqz_layer)
			exp1_layer = Activation( 'relu', name=s_id+relu+exp1)(exp1_layer)
			#3*3
			exp3_layer = Conv2D( expand, kernel_size=(3,3), padding='same', name=s_id+exp3)(sqz_layer)
			exp3_layer = Activation( 'relu', name=s_id+relu+exp3)(exp3_layer)

			cnct_layer = concatenate([exp1_layer, exp3_layer])

			return cnct_layer

		def squeezeNet():
			
			inputs = Input(shape=img_shape)

			x = Conv2D(96, kernel_size=(4,4), padding='same', name='conv1' )(inputs)
			x = Activation('relu', name='relu_conv1')(x)
			x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool1')(x)

			x = fire_module(x, fire_id=2, squeeze=16, expand=64)
			x = fire_module(x, fire_id=3, squeeze=16, expand=64)
			x = fire_module(x, fire_id=4, squeeze=32, expand=128)		
			x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool2')(x)

			x = fire_module(x, fire_id=5, squeeze=32, expand=128)
			x = fire_module(x, fire_id=6, squeeze=48, expand=192)
			x = fire_module(x, fire_id=7, squeeze=48, expand=192)
			x = fire_module(x, fire_id=8, squeeze=64, expand=256)
			x = MaxPool2D(pool_size=(3,3), strides=(2,2), name='pool3')(x)

			x = fire_module(x, fire_id=9, squeeze=64, expand=256)
			x = BatchNormalization()(x)
			x = Conv2D(10, kernel_size=(4,4), padding='same', name='conv10')(x)
			x = Activation('relu', name='relu_conv10')(x)

			x = GlobalAveragePooling2D()(x)
			model = Model(
				inputs = inputs,
				outputs = x,
				name = 'squeezeNet'
				)
			
			return model
		
		input_base = Input(shape=img_shape)
		input_pair = Input(shape=img_shape)

		basemodel = squeezeNet()
		encode_base = basemodel(input_base)
		encode_pair = basemodel(input_pair)

		L2_layer = Lambda( lambda tensor: K.sqrt(K.sum((tensor[0]-tensor[1])**2, axis=1, keepdims=True )),  output_shape=eucl_dist_output_shape)
		L2_distance = L2_layer([encode_base, encode_pair])

		model = Model(
				inputs = [input_base, input_pair],
				outputs= L2_distance,
				name='style_checker'
				)

		return model

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
			x = LeakyReLU(alpha=relu_alpha)(x)
			x = Dropout(dropout_rate, name=res_id+'_dropout_1')(x)

			x = Conv2D(second_channels, kernel_size=(kernel_length,kernel_length), strides=(1,1), padding='same', name=res_id+'_conv_2')(x)
			
			shortcut = shortcutModule(pre_lyr, second_channels, res_id, is_stride)
			x = concatenate([x, shortcut], name=res_id+'_conc')
			x = BatchNormalization(name=res_id+'_batch_2')(x)
			x = LeakyReLU(alpha=relu_alpha)(x)
			x = Dropout(dropout_rate, name=res_id+'_dropout_2')(x)
			return x

		def encoder(pre_lyr, name=''):
			x = Conv2D(gen_filter_num, kernel_size=(3,3), strides=(1,1), padding='same', name= name+'_encoder_conv_1')(pre_lyr)
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
			x = Conv2DTranspose( 2 * gen_filter_num, kernel_size=(4,4), strides=(2,2), padding='same', name=name+'_decoder_conv_1')(pre_lyr)
			x = BatchNormalization(name= name+'_decoder_norm_1')(x)
			x = LeakyReLU(alpha=relu_alpha, name= name+'_decoder_relu_1')(x)

			x = Conv2DTranspose( gen_filter_num, kernel_size=(4,4), strides=(2,2), padding='same', name=name+'_decoder_conv_2')(x)
			x = BatchNormalization(name= name+'_decoder_norm_2')(x)
			x = LeakyReLU(alpha=relu_alpha, name= name+'_decoder_relu_2')(x)

			if use_tanh:
				x = Conv2DTranspose( img_depth, kernel_size=(4,4), activation='tanh' ,strides=(1,1), padding='same', name=name+'_decoder_conv_3')(x)
			else:
				x = Conv2DTranspose( img_depth, kernel_size=(4,4), activation='sigmoid' ,strides=(1,1), padding='same', name=name+'_decoder_conv_3')(x)

			return x

		img = Input(shape=img_shape)

		x1 = encoder(img, name=name)
		# x1 = GaussianNoise(0.1)(x1)
		x2 = transfer(x1, name=name)
		output_gen = decoder(x2, name=name)

		model = Model(
			inputs = img,
			outputs = output_gen)

		return model

	def discriminator(self, name = ''):

		def d_layer(pre_lyr, filters, kernel_size=(3,3), name=''):
			x = Conv2D(filters, kernel_size=kernel_size, strides=(2,2), padding='same', name= 'disc_conv_'+ name)(pre_lyr)
			x = BatchNormalization(name= 'disc_norm_'+ name)(x)
			x = LeakyReLU(alpha=relu_alpha, name= 'disc_relu_'+ name)(x)
			x = Dropout(dropout_rate, name='disc_dropout_' + name)(x)
			return x

		img = Input(shape=img_shape)

		x1 = d_layer(img, dis_filter_num, kernel_size=(5,5), name='1')
		x2 = d_layer(x1, 2 * dis_filter_num, kernel_size=(5,5), name='2')
		# The discriminator is too strong
		x3 = d_layer(x2, 4 * dis_filter_num, kernel_size=(3,3), name='3')
		x4 = d_layer(x3, 4 * dis_filter_num, kernel_size=(3,3), name='4')

		prediction = Conv2D( 1 ,kernel_size=(4,4), activation='sigmoid' ,padding='same', name= name+'_disc_conv_pred')(x4)
		model = Model(
			inputs = img,
			outputs = prediction
		)
		return model


	def train(self, epochs, batch_size = 1, sample_interval = 2000):
		start_time = datetime.datetime.now()

		# Adversarial loss ground truths
		# shape = (1, 3, 3, 1)
		valid = np.ones((batch_size, ) + self.disc_patch )
		fake = np.zeros((batch_size, ) + self.disc_patch )
		ones = np.ones((batch_size))
		zeros = np.zeros((batch_size))

		train_discriminator = True

		long_valid = np.ones((2 * batch_size, ) + self.disc_patch )

		for epoch in range(epochs):
			acc_sum = 0
			d_loss_sum = 0
			g_loss_sum = 0 
			s_loss_sum = 0
			g_adv_loss_sum = 0
			g_con_loss_sum = 0
			id_loss_sum = 0

			for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
				# Train Discriminator
				fake_B = self.g_AtoB.predict(imgs_A)
				fake_A = self.g_BtoA.predict(imgs_B)

				recon_A = self.g_BtoA.predict(fake_B)
				recon_B = self.g_AtoB.predict(fake_A)

				# noise = np.random.normal(0, 0.5, imgs_A.shape)
				# noise_A = imgs_A + noise
				# noise_B = imgs_B + noise

				if train_discriminator:

					if disc_slower:
						mix_A = np.concatenate((imgs_A, recon_A),axis= 0 )
						mix_B = np.concatenate((imgs_B, recon_B),axis= 0 )
						dA_loss_real = self.d_A.train_on_batch(mix_A, long_valid)
						dB_loss_real = self.d_B.train_on_batch(mix_B, long_valid)
					else:
						dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
						dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)

					dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
					dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

					dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
					dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

					d_loss = 0.5 * np.add(dA_loss, dB_loss)

				if use_style:
					g_loss = self.combine.train_on_batch([imgs_A, imgs_B],
													[valid, valid,
													imgs_A, imgs_B,
													imgs_A, imgs_B,
													zeros, zeros])
					s_loss_sum += np.mean(g_loss[7:9])
				elif use_autoencoder:
					encoding_A = self.auto_encoder.predict(imgs_A)[1]
					encoding_B = self.auto_encoder.predict(imgs_B)[1]
					g_loss = self.combine.train_on_batch([imgs_A, imgs_B],
													[valid, valid,
													imgs_A, imgs_B,
													imgs_A, imgs_B,
													encoding_A, encoding_B])
				else:
					g_loss = self.combine.train_on_batch([imgs_A, imgs_B],
													[valid, valid,
													imgs_A, imgs_B,
													imgs_A, imgs_B])
				acc_sum += 100 * d_loss[1]
				d_loss_sum += d_loss[0]
				g_loss_sum += g_loss[0]
				g_adv_loss_sum += np.mean(g_loss[1:3])
				g_con_loss_sum += np.mean(g_loss[3:5])
				id_loss_sum += np.mean(g_loss[5:7])
				
				
				if (batch_i + 1) % sample_interval == 0:

					elapsed_time = datetime.datetime.now() - start_time
					print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%][G loss: %05f, adv: %05f, recon: %05f, id:%05f] time: %s " \
														% ( epoch, epochs,
															batch_i, self.data_loader.n_batches,
															d_loss[0], 100 * d_loss[1],
															g_loss[0],
															np.mean(g_loss[1:3]),
															np.mean(g_loss[3:5]),
															np.mean(g_loss[5:6]),
															elapsed_time))
					self.sample_images(epoch, batch_i)

			elapsed_time = datetime.datetime.now() - start_time
			print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%][G loss: %05f, adv: %05f, recon: %05f, id:%05f, style:%05f] time: %s " \
												% ( epoch, epochs,
													d_loss_sum/self.data_loader.n_batches, acc_sum/self.data_loader.n_batches,
													g_loss_sum/self.data_loader.n_batches,
													g_adv_loss_sum/self.data_loader.n_batches,
													g_con_loss_sum/self.data_loader.n_batches,
													id_loss_sum/self.data_loader.n_batches,
													s_loss_sum/self.data_loader.n_batches,
													elapsed_time))
			self.sample_images(epoch, self.data_loader.n_batches)

			if acc_sum / self.data_loader.n_batches < 90:
				train_discriminator = True
			elif g_adv_loss_sum / d_loss_sum < dg_ratio:
				train_discriminator = True
			else:
				train_discriminator = False

			

		self.store_model('models')

	def draw_result(self):
		os.makedirs('images/%s' % self.dataset_name, exist_ok=True)

		imgs_A, imgs_B = self.data_loader.load_test()
		fake_B = self.g_AtoB.predict(imgs_A)
		fake_A = self.g_AtoB.predict(imgs_B)

		imgs_A = 0.5 * imgs_A + 0.5
		imgs_B = 0.5 * imgs_B + 0.5
		fake_A = 0.5 * fake_A + 0.5
		fake_B = 0.5 * fake_B + 0.5

		print('imgs_A.shape')
		print(imgs_A.shape)

		r , c = len(imgs_A), 3

		titles = ['Source', 'Generated', 'Truth']
		fig, axs = plt.subplots(r, c)
		
		for i in range(r):
			display = imgs_A[i].reshape(imgs_A[i].shape[0], imgs_A[i].shape[1]) 
			axs[i,0].imshow(display, cmap='gray')
			axs[i,0].axis('off')

			display = fake_B[i].reshape(fake_B[i].shape[0], fake_B[i].shape[1]) 
			axs[i,1].imshow(display, cmap='gray')
			axs[i,1].axis('off')

			display = imgs_B[i].reshape(imgs_B[i].shape[0], imgs_B[i].shape[1]) 
			axs[i,2].imshow(display, cmap='gray')
			axs[i,2].axis('off')


		for ax, title in zip(axs[0], titles):
			ax.set_title(title)

		fig.savefig("images/%s/result.png" % (self.dataset_name))
		plt.close()

		r , c = len(imgs_B), 3

		titles = ['Source', 'Generated', 'Truth']
		fig, axs = plt.subplots(r, c)
		
		for i in range(r):
			display = imgs_B[i].reshape(imgs_B[i].shape[0], imgs_B[i].shape[1]) 
			axs[i,0].imshow(display, cmap='gray')
			axs[i,0].axis('off')

			display = fake_A[i].reshape(fake_A[i].shape[0], fake_A[i].shape[1]) 
			axs[i,1].imshow(display, cmap='gray')
			axs[i,1].axis('off')

			display = imgs_A[i].reshape(imgs_A[i].shape[0], imgs_A[i].shape[1]) 
			axs[i,2].imshow(display, cmap='gray')
			axs[i,2].axis('off')


		for ax, title in zip(axs[0], titles):
			ax.set_title(title)

		fig.savefig("images/%s/result2.png" % (self.dataset_name))
		plt.close()

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
		if use_tanh:
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


	def store_model(self, path):
		g_AtoB_h5_storage_path = path + '/ga2b.h5'
		g_BtoA_h5_storage_path = path + '/gb2a.h5'
		combine_h5_storage_path = path + '/combine.h5'

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

		save_model(
			self.combine,
			combine_h5_storage_path,
			overwrite = True,
			include_optimizer=True
		)


	def load_model(self, path):
		g_AtoB_h5_storage_path = path + '/ga2b.h5'
		g_BtoA_h5_storage_path = path + '/gb2a.h5'
		combine_h5_storage_path = path + '/combine.h5'

		d_A_h5_storage_path = path + '/da.h5'
		d_B_h5_storage_path = path + '/db.h5'
		error = False
		try:
			self.g_AtoB = load_model(
				g_AtoB_h5_storage_path,
				custom_objects={'g_optimizer':self.g_optimizer},
				compile=False
			)

			self.g_BtoA = load_model(
				g_BtoA_h5_storage_path,
				custom_objects={'g_optimizer':self.g_optimizer},
				compile=False
			)

			self.d_A = load_model(
				d_A_h5_storage_path,
				custom_objects={'d_optimizer':self.d_optimizer},
				compile=False
			)

			self.d_B = load_model(
				d_B_h5_storage_path,
				custom_objects={'d_optimizer':self.d_optimizer},
				compile=False
			)

			self.combine = load_model(
				combine_h5_storage_path,
				custom_objects={'g_optimizer':self.g_optimizer},
				compile=False
			)

			self.d_A.compile(
				loss = 'mse',
				optimizer = self.d_optimizer,
				metrics = ['accuracy']
				)
			self.d_B.compile(
				loss = 'mse',
				optimizer = self.d_optimizer,
				metrics = ['accuracy']
				)

			self.combine.compile(
							loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
							loss_weights = [1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id],
							optimizer = self.g_optimizer
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
		tf.set_random_seed(1201)
		gan = CycleGAN()
		if gan.load_model('models'):
			gan.train(epochs=epochs, batch_size = batch_size,  sample_interval=sample_interval)
		else:
			gan.draw_result()

