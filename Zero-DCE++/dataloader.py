import os
import sys

import torch
import torch.utils.data as tud

from PIL import Image
import numpy as np
import glob
import random


random.seed(1143)


def populate_train_list(lowlight_images_path):
	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	train_list = image_list_lowlight
	random.shuffle(train_list)
	return train_list


class lowlight_loader(tud.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_list = populate_train_list(lowlight_images_path) 
		self.size = 512

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))

	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		
		data_lowlight = Image.open(data_lowlight_path)
		
		data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)


class StructuredLoader(tud.Dataset):
	"""
	This is a data loader designed to work with my Folder Structure.
	It assumes that images are numbered sequentially from 0 up to some maximum number
	"""
	def __init__(self, images, scaling, downsampling, fmt='*.jpg'):
		"""
		Initialise the DataSet

		:param images: Path to Root Directory
		"""
		# Initialise Directory Structure
		self.__images_pths = [
			img for pth in os.listdir(images) for img in glob.glob(os.path.join(images, pth, fmt))[::downsampling]
		]
		self.__scaling = scaling

	def __getitem__(self, index):
		# Get Image
		image = Image.open(self.__images_pths[index])
		image = np.asarray(image) / 255.0  # .resize((self.RESCALE, self.RESCALE), Image.ANTIALIAS)

		# Compute Sizes to get common size
		i_size = np.asarray(image.shape[:2])
		o_h, o_w = (i_size // self.__scaling) * self.__scaling
		s_h, s_w = (i_size - (o_h, o_w)) // 2
		image = image[s_h:s_h + o_h, s_w:s_w + o_w, :]

		# Convert to Torch and return permuted
		image = torch.from_numpy(image).float()
		return image.permute(2, 0, 1)

	def __len__(self):
		return len(self.__images_pths)
