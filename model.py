import tensorflow as tf
import pandas as pd
import os
import numpy as np
import sys
from matplotlib import pyplot as plt

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed

from keras.models import Model
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

sys.path.add('/content/drive/My Drive/Colab Notebooks/')

def vgg16(input_tensor = None, trainable = False):

	inputImg = Input(shape = (None,None,3))

	x = Conv2D(64, (3,3), activation = 'relu', padding = 'same', name = 'block1_conv1')(inputImg)
	x = Conv2D(64, (3,3), activation = 'relu', padding = 'same', name = 'block1_conv2')(x)
	x = MaxPooling2D((2,2), strides = (2,2), name='block1_pool')(x)

	x = Conv2D(128, (3,3), activation = 'relu', padding = 'same', name = 'block2_conv1')(x)
	x = Conv2D(128, (3,3), activation = 'relu', padding = 'same', name = 'block2_conv2')(x)
	x = MaxPooling2D((2,2), strides = (2,2), name='block2_pool')(x)

	x = Conv2D(256,(3,3), activation = 'relu', padding = 'same', name = 'block3_conv1')(x)
	x = Conv2D(256,(3,3), activation = 'relu', padding = 'same', name = 'block3_conv2')(x)
	x = Conv2D(256,(3,3), activation = 'relu', padding = 'same', name = 'block3_conv3')(x)
	x = MaxPooling2D((2,2), strides = (2,2), name='block3_pool')(x)

	x = Conv2D(512,(3,3), activation = 'relu', padding = 'same', name = 'block4_conv1')(x)
	x = Conv2D(512,(3,3), activation = 'relu', padding = 'same', name = 'block4_conv2')(x)
	x = Conv2D(512,(3,3), activation = 'relu', padding = 'same', name = 'block4_conv3')(x)
	x = MaxPooling2D((2,2), strides = (2,2), name='block4_pool')(x)

	x = Conv2D(512,(3,3), activation = 'relu', padding = 'same', name = 'block5_conv1')(x)
	x = Conv2D(512,(3,3), activation = 'relu', padding = 'same', name = 'block5_conv2')(x)
	x = Conv2D(512,(3,3), activation = 'relu', padding = 'same', name = 'block5_conv3')(x)
	
	return x

def rpnLayer(baseLayer,anchorNum = 9):

	x = Conv2D(512, (3,3), activation = 'relu', padding= 'same', kernel_initializer='normal', name = 'rpn_conv1')(baseLayer)

	biclassX = Conv2D(anchorNum, (1,1), activation = 'sigmoid', kernel_initializer='uniform', name = 'rpn_out_class')(x)

	regrX = Conv2D(anchorNum * 4, (1,1), activation = 'linear', kernel_initializer='zero', name = 'rpn_out_regress')(x)

	return [biclassX,regrX,baseLayer]

class roiPooling(Layer):
	
	def __init__(self, pool_size, num_rois, **kwargs):
		
		self.dim_ordering = K.image_dim_ordering()
		self.pool_size = pool_size
		self.num_rois = num_rois

		super(RoiPoolingConv, self).__init__(**kwargs)

	def build(self, input_shape):
		self.nb_channels = input_shape[0][3]

	def compute_output_shape(self, input_shape):
		return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

	def call(self, x, maxk = None):
		img = x[0]
		roiList = x[1]

		input_shape = K.shape(img)
		resizeRoIList = []

		for index in range(self.num_rois):
			x = roiList[0, index, 0]
			y = roiList[0, index, 1]
			w = roiList[0, index, 2]
			h = roiList[0, index, 3]

			x = K.cast(x, 'int32')
			y = K.cast(y, 'int32')
			w = K.cast(w, 'int32')
			h = K.cast(h, 'int32')

			resizeRoI = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
			resizeRoIList.append(resizeRoI)

		output = K.concatenate(resizeRoIList, axis=0)
		output = K.reshape(output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
		finalOutput = K.permute_dimensions(output, (0, 1, 2, 3, 4))

		return finalOutput

	def get_config(self):
		config = {'pool_size': self.pool_size,'num_rois': self.num_rois}
		base_config = super(RoiPoolingConv, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


def classifier(baseLayer, RoI, roiNum, classNum = 7):

	pool_size = 7

	roiPoolingResult = RoiPoolingConv(pool_size,roiNum)([baseLayer, RoI])

	x = TimeDistributed(Flatten(name = 'flatten'))(roiPoolingResult)

	x = TimeDistributed(Dense(4096, activation = 'relu', name = 'fc1'))(x)
	x = TimeDistributed(Dropout(0.5))(x)

	x = TimeDistributed(Dense(4096, activation = 'relu', name = 'fc2'))(x)
	result = TimeDistributed(Dropout(0.5))(x)

	resultClass = TimeDistributed(Dense(classNum, activation = 'softmax', kernel_initializer = 'zero'), name = 'dense_class_{}'.format(classNum))(result)

	resultRegr = TimeDistributed(Dense(4 * (classNum - 1), activation = 'linear', kernel_initializer = 'zero'), name = 'dense_regress_{}'.format(classNum))(result)

	return [resultClass,resultRegr]

def rpn_loss_regr(num_anchors):
	
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, :, 4 * num_anchors:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
		return 1.0 * K.sum(
			y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(1e-4 + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):

	def rpn_loss_cls_fixed_num(y_true, y_pred):
			return 1.0 * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(1e-4 + y_true[:, :, :, :num_anchors])

	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return 1.0 * K.sum(y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(1e-4 + y_true[:, :, :4 * num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	return 1.0 * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

'''
baseLayer = vgg16(trainable = True)
rpn = rpnLayer(baseLayer)
'''

