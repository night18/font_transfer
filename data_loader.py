import scipy
from glob import glob
import numpy as np
import random


class DataLoader():
	def __init__(self, dataset_name, img_res=(128, 128), use_tanh=True):
		self.dataset_name = dataset_name
		self.img_res = img_res
		self.use_tanh = use_tanh

	def load_data(self, domain, batch_size=1, is_testing=False, is_reshape = True):
		data_type = "train%s" % domain if not is_testing else "test%s" % domain
		path = glob('./%s/%s/*' % (self.dataset_name, data_type))

		batch_images = np.random.choice(path, size=batch_size)

		imgs = []
		for img_path in batch_images:
			img = self.imread(img_path)
			if not is_testing:
				img = scipy.misc.imresize(img, self.img_res)

				# if np.random.random() > 0.5:
				# 	img = np.fliplr(img)
			else:
				img = scipy.misc.imresize(img, self.img_res)
			imgs.append(img)

		imgs = np.array(imgs)
		if is_reshape:
			imgs = imgs.reshape(imgs.shape[0], self.img_res[0], self.img_res[1], 1)
		else:
			imgs = imgs.reshape(self.img_res[0], self.img_res[1], 1) 
			
		if self.use_tanh:	
			imgs = imgs/127.5 - 1.
		else: 
			imgs = imgs/255.0

		return imgs

	def load_all_img(self):
		path_A = sorted(glob('./%s/%sA/*' % (self.dataset_name, 'train')))
		path_B = sorted(glob('./%s/%sB/*' % (self.dataset_name, 'train')))

		imgs = []
		for i in range(len(path_A)):
			img_A = self.imread(path_A[i])	
			img_A = scipy.misc.imresize(img_A, self.img_res)
			imgs.append(img_A)

		for i in range(len(path_B)):
			img_B = self.imread(path_B[i])	
			img_B = scipy.misc.imresize(img_B, self.img_res)
			imgs.append(img_B)

		imgs = np.array(imgs)
		imgs = imgs.reshape(imgs.shape[0], self.img_res[0], self.img_res[1], 1)
		imgs = imgs/127.5 -1.
		return imgs


	def load_test(self):
		path_A = sorted(glob('./%s/%sA/*' % (self.dataset_name, 'test')))
		path_B = sorted(glob('./%s/%sB/*' % (self.dataset_name, 'test')))
		imgs_A, imgs_B = [], []

		for i in range(len(path_A)):
			img_A = self.imread(path_A[i])	
			img_B = self.imread(path_B[i])

			img_A = scipy.misc.imresize(img_A, self.img_res)
			img_B = scipy.misc.imresize(img_B, self.img_res)

			imgs_A.append(img_A)
			imgs_B.append(img_B)

		imgs_A = np.array(imgs_A)
		imgs_B = np.array(imgs_B)

		imgs_A = imgs_A.reshape(imgs_A.shape[0], self.img_res[0], self.img_res[1], 1)
		imgs_B = imgs_B.reshape(imgs_B.shape[0], self.img_res[0], self.img_res[1], 1)

		imgs_A = imgs_A/127.5 - 1.
		imgs_B = imgs_B/127.5 - 1.

		return imgs_A, imgs_B

			


	def load_batch(self, batch_size=1, is_testing=False):
		data_type = "train" if not is_testing else "val"
		path_A = glob('./%s/%sA/*' % (self.dataset_name, data_type))
		path_B = glob('./%s/%sB/*' % (self.dataset_name, data_type))

		self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
		total_samples = self.n_batches * batch_size

		# Sample n_batches * batch_size from each path list so that model sees all
		# samples from both domains
		path_A = np.random.choice(path_A, total_samples, replace=False)
		path_B = np.random.choice(path_B, total_samples, replace=False)

		for i in range(self.n_batches-1):
			batch_A = path_A[i*batch_size:(i+1)*batch_size]
			batch_B = path_B[i*batch_size:(i+1)*batch_size]
			imgs_A, imgs_B = [], []
			for img_A, img_B in zip(batch_A, batch_B):
				img_A = self.imread(img_A)
				img_B = self.imread(img_B)

				img_A = scipy.misc.imresize(img_A, self.img_res)
				img_B = scipy.misc.imresize(img_B, self.img_res)

				'''
				if not is_testing and np.random.random() > 0.5:
						img_A = np.fliplr(img_A)
						img_B = np.fliplr(img_B)
				'''

				imgs_A.append(img_A)
				imgs_B.append(img_B)


			imgs_A = np.array(imgs_A)
			imgs_B = np.array(imgs_B)


			imgs_A = imgs_A.reshape(imgs_A.shape[0], self.img_res[0], self.img_res[1], 1)
			imgs_B = imgs_B.reshape(imgs_B.shape[0], self.img_res[0], self.img_res[1], 1)


			if self.use_tanh:
				# range (-1, 1)
				imgs_A = imgs_A/127.5 - 1.
				imgs_B = imgs_B/127.5 - 1.
			else:
				imgs_A = imgs_A/255.0
				imgs_B = imgs_B/255.0

			yield imgs_A, imgs_B

	def load_img(self, path):
		img = self.imread(path)
		img = scipy.misc.imresize(img, self.img_res)
		if self.use_tanh:
			img = img/127.5 - 1.
		else:
			img = img/255.0

		return img[np.newaxis, :, :, :]

	def imread(self, path):
		return scipy.misc.imread(path, mode='RGB', flatten=True).astype(np.float)
		# return scipy.misc.imread(path, mode='RGB').astype(np.float)


	def create_style_siamese(self):
		train_idx_label = []

		for x in range(2000):
			base_A = self.load_data(domain='A', batch_size=1, is_testing=False, is_reshape = False)
			base_B = self.load_data(domain='B', batch_size=1, is_testing=False, is_reshape = False)
			pair_A = self.load_data(domain='A', batch_size=1, is_testing=False, is_reshape = False)
			pair_B = self.load_data(domain='B', batch_size=1, is_testing=False, is_reshape = False)

			true_AA = (base_A, pair_A, 0)
			true_BB = (base_B, pair_B, 0)
			False_AB = (base_A, pair_B, 1)
			False_BA = (base_B, pair_A, 1)

			train_idx_label.append(true_AA)
			train_idx_label.append(true_BB)
			train_idx_label.append(False_AB)
			train_idx_label.append(False_BA)
		
		random.shuffle(train_idx_label)
		return train_idx_label