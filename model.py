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

gen_filter_num = 32
dis_filter_num = 64
batch_size = 1
pool_size = 50
img_width = 256
img_height = 256
img_depth = 3
relu_alpha = 0.05
dropout_rate = 0.2
learning_rate = 0.01

class CycleGAN():
	def __init__(self):
		# Todo: Load data
		optimizers = Adam(learning_rate)
		
		#Build and compile the discriminator
		self.d_A = self.discriminator(name='d_A')
		self.d_B = 
		# Chun


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
		x = LeakyReLU(alpha=0.relu_alpha)(x)

		x = Conv2D(second_channels, kernel_size=(kernel_length,kernel_length), strides=(1,1), padding='same', name=res_id+'_conv_2')(x)
		
		shortcut = shortcutModule(pre_lyr, second_channels, res_id, is_stride)
		x = concatenate([x, shortcut], name=res_id+'_conc')
		x = BatchNormalization(name=res_id+'_batch_2')(x)
		x = Dropout(dropout_rate, name=res_id+'_dropout_2')(x)
		x = LeakyReLU(alpha=0.relu_alpha)(x)
		return x


	def generator(self, name = ''):
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


		img = Input(shape=(img_height, img_width, img_depth))

		x1 = encoder(img, name=name)
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

		img = Input(shape=(img_height, img_width, img_depth))

		x1 = d_layer(img, dis_filter_num, kernel_size=(5,5), name='1')
		x2 = d_layer(x1, 2 * dis_filter_num, kernel_size=(5,5), name='2')
		x3 = d_layer(x2, 4 * dis_filter_num, kernel_size=(3,3), name='3')
		x4 = d_layer(x3, 4 * dis_filter_num, kernel_size=(3,3), name='3')

		prediction = Dense( 1, activation='sigmoid' , padding='same', name= name+'_disc_conv_pred')(x4)
		model = Model(
			inputs = img,
			outputs = prediction
		)
		return model



	# discriminator loss
	# if the input is ground truth, the prediction should be close to 1(True)(yTrue = 1)
	# if the input is generative, the prediction should be close to 0(False)(yTrue = 0)
	def net_d_loss(yTrue, yPred):
		return K.mean( K.square(yTrue - yPred) )

	# generator should create the image which is similar to the groud truth, i.e., dis_gen_Chinese should be 1
	def net_g_loss(yTrue, yPred):
		return K.mean( K.square(1 - yPred) )

	def cycle_loss(yTrue, yPred):
		return K.mean (K.abs(yTrue, yPred) )