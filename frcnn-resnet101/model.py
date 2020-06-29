import tensorflow as tf
import pandas as pd
import os
import numpy as np
import sys
from matplotlib import pyplot as plt
import random
import time

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Add, Activation, ZeroPadding2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed, AveragePooling2D, BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
sys.path.append('/content/drive/My Drive/Colab Notebooks/')
import dataProcessor
import dataGenerator

class FixedBatchNormalization(Layer):

	def __init__(self, epsilon=1e-3, axis=-1,
				 weights=None, beta_init='zero', gamma_init='one',
				 gamma_regularizer=None, beta_regularizer=None, **kwargs):

		self.supports_masking = True
		self.beta_init = initializers.get(beta_init)
		self.gamma_init = initializers.get(gamma_init)
		self.epsilon = epsilon
		self.axis = axis
		self.gamma_regularizer = regularizers.get(gamma_regularizer)
		self.beta_regularizer = regularizers.get(beta_regularizer)
		self.initial_weights = weights
		super(FixedBatchNormalization, self).__init__(**kwargs)

	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		shape = (input_shape[self.axis],)

		self.gamma = self.add_weight(shape = shape,
									 initializer=self.gamma_init,
									 regularizer=self.gamma_regularizer,
									 name='{}_gamma'.format(self.name),
									 trainable=False)
		self.beta = self.add_weight(shape = shape,
									initializer=self.beta_init,
									regularizer=self.beta_regularizer,
									name='{}_beta'.format(self.name),
									trainable=False)
		self.running_mean = self.add_weight(shape = shape, initializer='zero',
											name='{}_running_mean'.format(self.name),
											trainable=False)
		self.running_std = self.add_weight(shape = shape, initializer='one',
										   name='{}_running_std'.format(self.name),
										   trainable=False)

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

		self.built = True

	def call(self, x, mask=None):

		assert self.built, 'Layer must be built before being called'
		input_shape = K.int_shape(x)

		reduction_axes = list(range(len(input_shape)))
		del reduction_axes[self.axis]
		broadcast_shape = [1] * len(input_shape)
		broadcast_shape[self.axis] = input_shape[self.axis]

		if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
			x_normed = K.batch_normalization(
				x, self.running_mean, self.running_std,
				self.beta, self.gamma,
				epsilon=self.epsilon)
		else:
			# need broadcasting
			broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
			broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
			broadcast_beta = K.reshape(self.beta, broadcast_shape)
			broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
			x_normed = K.batch_normalization(
				x, broadcast_running_mean, broadcast_running_std,
				broadcast_beta, broadcast_gamma,
				epsilon=self.epsilon)

		return x_normed

	def get_config(self):
		config = {'epsilon': self.epsilon,
				  'axis': self.axis,
				  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
				  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None}
		base_config = super(FixedBatchNormalization, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, filters, stage, block, trainable = False):
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters[0], (1, 1), name = conv_name_base + '2a', trainable = trainable)(input_tensor)
	x = FixedBatchNormalization(axis = 3, name = bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters[1], (3, 3), padding = 'same', name = conv_name_base + '2b', trainable = trainable)(x)
	x = FixedBatchNormalization(axis = 3, name = bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters[2], (1, 1), name = conv_name_base + '2c', trainable = trainable)(x)
	x = FixedBatchNormalization(axis = 3, name = bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)
	return x

def conv_block(input_tensor, filters, stage, block, strides = (2, 2), trainable = False):
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters[0], (1, 1), strides = strides, name = conv_name_base + '2a', trainable = trainable)(input_tensor)
	x = FixedBatchNormalization(axis = 3, name = bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters[1], (3, 3), padding = 'same', name = conv_name_base + '2b', trainable = trainable)(x)
	x = FixedBatchNormalization(axis = 3, name = bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters[2], (1, 1), name = conv_name_base + '2c', trainable = trainable)(x)
	x = FixedBatchNormalization(axis = 3, name = bn_name_base + '2c')(x)

	x1 = Conv2D(filters[2], (1, 1), strides = strides, name = conv_name_base + '1', trainable = trainable)(input_tensor)
	x1 = FixedBatchNormalization(axis = 3, name = bn_name_base + '1')(x1)

	x = Add()([x, x1])
	x = Activation('relu')(x)
	return x

def identity_block_td(input_tensor, filters, stage, block):
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = TimeDistributed(Conv2D(filters[0], (1, 1), kernel_initializer = 'normal'), name = conv_name_base + '2a')(input_tensor)
	x = TimeDistributed(FixedBatchNormalization(axis = 3), name = bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(filters[1], (3, 3), padding = 'same', kernel_initializer = 'normal'), name = conv_name_base + '2b')(x)
	x = TimeDistributed(FixedBatchNormalization(axis = 3), name = bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(filters[2], (1, 1), kernel_initializer = 'normal'), name = conv_name_base + '2c')(x)
	x = TimeDistributed(FixedBatchNormalization(axis = 3), name = bn_name_base + '2c')(x)

	x = Add()([x, input_tensor])
	x = Activation('relu')(x)
	return x

def conv_block_td(input_tensor, filters, stage, block, strides = (2, 2)):
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'
	
	x = TimeDistributed(Conv2D(filters[0], (1, 1), strides = strides, kernel_initializer = 'normal'), input_shape = (8, 14, 14, 1024), name = conv_name_base + '2a')(input_tensor)
	x = TimeDistributed(FixedBatchNormalization(axis = 3), name = bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(filters[1], (3, 3), padding = 'same', kernel_initializer = 'normal'), name = conv_name_base + '2b')(x)
	x = TimeDistributed(FixedBatchNormalization(axis = 3), name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = TimeDistributed(Conv2D(filters[2], (1, 1), kernel_initializer = 'normal'), name = conv_name_base + '2c')(x)
	x = TimeDistributed(FixedBatchNormalization(axis = 3), name = bn_name_base + '2c')(x)

	x1 = TimeDistributed(Conv2D(filters[2], (1, 1), strides = strides, kernel_initializer = 'normal'), name = conv_name_base + '1')(input_tensor)
	x1 = TimeDistributed(FixedBatchNormalization(axis = 3), name = bn_name_base + '1')(x1)

	x = Add()([x, x1])
	x = Activation('relu')(x)
	return x

def resNet101(input_tensor = None,trainable = False):

	inputImg = input_tensor
	x = ZeroPadding2D((3, 3))(inputImg)
	
	x = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(x)
	x = FixedBatchNormalization(axis = 3, name = 'bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides = (2, 2))(x)

	x = conv_block(x, [64, 64, 256], 2, 'a', strides = (1, 1))
	x = identity_block(x, [64, 64, 256], 2, 'b')
	x = identity_block(x, [64, 64, 256], 2, 'c')

	x = conv_block(x, [128, 128, 512], 3, 'a')
	x = identity_block(x, [128, 128, 512], 3, 'b')
	x = identity_block(x, [128, 128, 512], 3, 'c')
	x = identity_block(x, [128, 128, 512], 3, 'd')

	x = conv_block(x, [256, 256, 1024], 4, 'a')
	x = identity_block(x, [256, 256, 1024], 4, 'b')
	x = identity_block(x, [256, 256, 1024], 4, 'c')
	x = identity_block(x, [256, 256, 1024], 4, 'd')
	x = identity_block(x, [256, 256, 1024], 4, 'e')
	x = identity_block(x, [256, 256, 1024], 4, 'f')
	x = identity_block(x, [256, 256, 1024], 4, 'g')
	x = identity_block(x, [256, 256, 1024], 4, 'h')
	x = identity_block(x, [256, 256, 1024], 4, 'i')
	x = identity_block(x, [256, 256, 1024], 4, 'j')
	x = identity_block(x, [256, 256, 1024], 4, 'k')
	x = identity_block(x, [256, 256, 1024], 4, 'l')
	x = identity_block(x, [256, 256, 1024], 4, 'm')
	x = identity_block(x, [256, 256, 1024], 4, 'n')
	x = identity_block(x, [256, 256, 1024], 4, 'o')
	x = identity_block(x, [256, 256, 1024], 4, 'p')
	x = identity_block(x, [256, 256, 1024], 4, 'q')
	x = identity_block(x, [256, 256, 1024], 4, 'r')
	x = identity_block(x, [256, 256, 1024], 4, 's')
	x = identity_block(x, [256, 256, 1024], 4, 't')
	x = identity_block(x, [256, 256, 1024], 4, 'u')
	x = identity_block(x, [256, 256, 1024], 4, 'v')
	x = identity_block(x, [256, 256, 1024], 4, 'w')

	return x

def rpnLayer(baseLayer,anchorNum = 9):

	x = Conv2D(512, (3,3), activation = 'relu', padding= 'same', kernel_initializer='normal', name = 'rpn_conv1')(baseLayer)

	biclassX = Conv2D(anchorNum, (1,1), activation = 'sigmoid', kernel_initializer='uniform', name = 'rpn_out_class')(x)

	regrX = Conv2D(anchorNum * 4, (1,1), activation = 'linear', kernel_initializer='zero', name = 'rpn_out_regress')(x)

	return [biclassX,regrX,baseLayer]

class roiPooling(Layer):
	
	def __init__(self, pool_size, num_rois, **kwargs):
		self.pool_size = pool_size
		self.num_rois = num_rois

		super(roiPooling, self).__init__(**kwargs)

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

			resizeRoI = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
			resizeRoIList.append(resizeRoI)

		output = K.concatenate(resizeRoIList, axis=0)
		output = K.reshape(output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
		finalOutput = K.permute_dimensions(output, (0, 1, 2, 3, 4))

		return finalOutput

	def get_config(self):
		config = {'pool_size': self.pool_size,'num_rois': self.num_rois}
		base_config = super(roiPooling, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def classifierLayer(baseLayer, roiInput, roiNum = 8, classNum = 7):

	pool_size = 14

	roiPoolingResult = roiPooling(pool_size,roiNum)([baseLayer, roiInput])

	x = conv_block_td(roiPoolingResult, [512, 512, 2048], 5, 'a', strides = (2, 2))
	x = identity_block_td(x, [512, 512, 2048], 5, 'b')
	x = identity_block_td(x, [512, 512, 2048], 5, 'c')
	x = TimeDistributed(AveragePooling2D((7, 7)), name = 'avg_pool')(x)

	result = TimeDistributed(Flatten(name='flatten'))(x)

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

def trainModel():
	
	imgList = dataProcessor.readJson()
	random.seed(1)
	random.shuffle(imgList)
	trainData = dataProcessor.getData(imgList)
	
	img_input = Input(shape=(None, None, 3))
	roi_input = Input(shape=(None, 4))

	anchorNum = 9
	classNum = 7

	resNet = resNet101(input_tensor = img_input, trainable = True)
	rpn = rpnLayer(resNet)
	classifier = classifierLayer(resNet,roi_input)

	model_rpn = Model(img_input, rpn[:2])
	model_classifier = Model([img_input,roi_input], classifier)
	model_all = Model([img_input,roi_input], rpn[:2] + classifier)

	optimizer = Adam(lr = 1e-5)
	optimizer_classifier = Adam(lr = 1e-5)
	model_rpn.compile(optimizer = optimizer, loss = [rpn_loss_cls(anchorNum), rpn_loss_regr(anchorNum)])
	model_classifier.compile(optimizer = optimizer_classifier, loss = [class_loss_cls, class_loss_regr(classNum - 1)], metrics={'dense_class_{}'.format(classNum): 'accuracy'})
	model_all.compile(optimizer = 'sgd', loss = 'mae')
	
	modelPath = '/content/drive/My Drive/Colab Notebooks/model/'
	record_path = modelPath + 'record_resnet.csv'

	if not os.path.isfile(modelPath + 'frcnn_resnet101.hdf5'):
		try:
			model_rpn.load_weights(modelPath + 'resnet101_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
			model_classifier.load_weights(modelPath + 'resnet101_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
			print('load success')
		except:
			print('load weights failed')
		record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
	else:
		model_rpn.load_weights(modelPath + 'frcnn_resnet101.hdf5', by_name=True)
		model_classifier.load_weights(modelPath + 'frcnn_resnet101.hdf5', by_name=True)

		record_df = pd.read_csv(record_path)

		r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
		r_class_acc = record_df['class_acc']
		r_loss_rpn_cls = record_df['loss_rpn_cls']
		r_loss_rpn_regr = record_df['loss_rpn_regr']
		r_loss_class_cls = record_df['loss_class_cls']
		r_loss_class_regr = record_df['loss_class_regr']
		r_curr_loss = record_df['curr_loss']
		r_elapsed_time = record_df['elapsed_time']
		r_mAP = record_df['mAP']

		print('Already train %dK batches'% (len(record_df)))

	num_epochs = 200
	epoch_length = 1000
	iter_num = 0
	total_epochs = len(record_df) + num_epochs
	r_epochs = len(record_df)

	losses = np.zeros((epoch_length, 5))

	rpn_accuracy_rpn_monitor = []
	rpn_accuracy_for_epoch = []

	if len(record_df)==0:
		best_loss = np.Inf
	else:
		best_loss = np.min(r_curr_loss)
	
	start_time = time.time()
	for epochNum in range(num_epochs):

		progbar = generic_utils.Progbar(epoch_length)
		print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))
		r_epochs += 1

		while True:
			try:
				if len(rpn_accuracy_rpn_monitor) == epoch_length:
					mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
					rpn_accuracy_rpn_monitor = []
					if mean_overlapping_bboxes == 0:
						print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

				X, Y, imgData = next(trainData)
				loss_rpn = model_rpn.train_on_batch(X,Y)
				proposal = model_rpn.predict_on_batch(X)
				roi = dataProcessor.proposalCreator(proposal[0], proposal[1])
				X2, Y1, Y2 = dataProcessor.roiHead(roi, imgData)
				if X2 is None:
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					continue
				selectSamples,posNum = dataProcessor.roiSelect(Y1)
				rpn_accuracy_rpn_monitor.append(posNum)
				rpn_accuracy_for_epoch.append(posNum)
				loss_class = model_classifier.train_on_batch([X, X2[:, selectSamples, :]], [Y1[:, selectSamples, :], Y2[:, selectSamples, :]])
				losses[iter_num, 0] = loss_rpn[1]
				losses[iter_num, 1] = loss_rpn[2]

				losses[iter_num, 2] = loss_class[1]
				losses[iter_num, 3] = loss_class[2]
				losses[iter_num, 4] = loss_class[3]

				iter_num += 1

				progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
									('final_cls', np.mean(losses[:iter_num, 2])), ('final_regr', np.mean(losses[:iter_num, 3]))])

				if iter_num == epoch_length:
					loss_rpn_cls = np.mean(losses[:, 0])
					loss_rpn_regr = np.mean(losses[:, 1])
					loss_class_cls = np.mean(losses[:, 2])
					loss_class_regr = np.mean(losses[:, 3])
					class_acc = np.mean(losses[:, 4])

					mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
					rpn_accuracy_for_epoch = []

					print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
					print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
					print('Loss RPN classifier: {}'.format(loss_rpn_cls))
					print('Loss RPN regression: {}'.format(loss_rpn_regr))
					print('Loss Detector classifier: {}'.format(loss_class_cls))
					print('Loss Detector regression: {}'.format(loss_class_regr))
					print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
					print('Elapsed time: {}'.format(time.time() - start_time))
					elapsed_time = (time.time()-start_time)/60

					curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
					iter_num = 0
					start_time = time.time()

					if curr_loss < best_loss:
						best_loss = curr_loss
						model_all.save_weights(modelPath + 'frcnn_resnet101.hdf5')

					new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3),
							'class_acc':round(class_acc, 3), 
							'loss_rpn_cls':round(loss_rpn_cls, 3), 
							'loss_rpn_regr':round(loss_rpn_regr, 3), 
							'loss_class_cls':round(loss_class_cls, 3), 
							'loss_class_regr':round(loss_class_regr, 3), 
							'curr_loss':round(curr_loss, 3), 
							'elapsed_time':round(elapsed_time, 3), 
							'mAP': 0}

					record_df = record_df.append(new_row, ignore_index=True)
					record_df.to_csv(record_path, index=0)

					break
			except Exception as e:
				print(e)
				continue
	print('Train Complete!')
	

def apply_regr(x, y, w, h, tx, ty, tw, th):
	try:
		cx = x + w/2.
		cy = y + h/2.
		cx1 = tx * w + cx
		cy1 = ty * h + cy
		w1 = math.exp(tw) * w
		h1 = math.exp(th) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.
		x1 = int(round(x1))
		y1 = int(round(y1))
		w1 = int(round(w1))
		h1 = int(round(h1))

		return x1, y1, w1, h1

	except ValueError:
		return x, y, w, h
	except OverflowError:
		return x, y, w, h
	except Exception as e:
		print(e)
		return x, y, w, h

def format_img_size(img):
	img_min_side = float(600)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img):
	#img = img[:, :, (2, 1, 0)]
	channelMean = [103.939, 116.779, 123.68]
	img = img.astype(np.float32)
	img[:, :, 0] -= channelMean[0]
	img[:, :, 1] -= channelMean[1]
	img[:, :, 2] -= channelMean[2]
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img):
	img, ratio = format_img_size(img)
	img = format_img_channels(img)
	return img, ratio

def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

def predictModel():
	regrStd = [8.0, 8.0, 4.0, 4.0]
	bbox_threshold = 0.7

	img_input = Input(shape = (None, None, 3))
	roi_input = Input(shape = (8, 4))
	feature_input = Input(shape = (None, None, 1024))

	resNet = resNet101(input_tensor = img_input, trainable = True)
	rpn = rpnLayer(resNet)
	classifier = classifierLayer(feature_input,roi_input)

	model_rpn = Model(img_input, rpn)
	model_classifier_only = Model([feature_input,roi_input], classifier)
	model_classifier = Model([feature_input,roi_input], classifier)

	modelPath = '/content/drive/My Drive/Colab Notebooks/model/'

	model_rpn.load_weights(modelPath + 'frcnn_resnet101.hdf5', by_name=True)
	model_classifier.load_weights(modelPath + 'frcnn_resnet101.hdf5', by_name=True)

	model_rpn.compile(optimizer='sgd', loss='mse')
	model_classifier.compile(optimizer='sgd', loss='mse')

	imgList = dataProcessor.readJson()
	random.seed(1)
	random.shuffle(imgList)
	imgList = imgList[:10]

	class_mapping = {'Person':0, 'Laptop':1, 'Cat':2, 'Mobile phone':3, 'Car':4, 'Human face':5, 'BG':6}
	class_mapping = {v: k for k, v in class_mapping.items()}
	class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

	for img in imgList:
		path = img['path']
		img = cv2.imread(img)
		X,ratio = format_img(img)
		X = np.transpose(X, (0, 2, 3, 1))
		proposal = model_rpn.predict(X)
		roi = dataProcessor.proposalCreator(proposal[0], proposal[1])
		roi[:, 2] -= roi[:, 0]
		roi[:, 3] -= roi[:, 1]

		bboxes = {}
		probs = {}

		for jk in range(roi.shape[0]//8 + 1):
			ROIs = np.expand_dims(roi[8 * jk:8 * (jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == roi.shape[0]//8:
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],8,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = model_classifier_only.predict([proposal[2], ROIs])
			# Calculate bboxes coordinates on resized image
			for ii in range(P_cls.shape[1]):
				# Ignore 'bg' class
				if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
					tx /= regrStd[0]
					ty /= regrStd[1]
					tw /= regrStd[2]
					th /= regrStd[3]
					x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([16*x, 16*y, 16*(x+w), 16*(y+h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		all_dets = []

		for key in bboxes:
			bbox = np.array(bboxes[key])

			new_boxes, new_probs = dataProcesspr.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
			for jk in range(new_boxes.shape[0]):
				print(new_boxes[jk,:])
				(x1, y1, x2, y2) = new_boxes[jk,:]

				# Calculate real coordinates on original image
				(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

				cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

				textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
				all_dets.append((key,100*new_probs[jk]))

				(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
				textOrg = (real_x1, real_y1-0)

				cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
				cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
				cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

		print(all_dets)
		plt.figure(figsize=(10,10))
		plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
		plt.show()