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
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
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

def vgg16(input_tensor=None,trainable = False):

	inputImg = input_tensor

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

def classifierLayer(baseLayer, roiInput, roiNum = 4, classNum = 7):

	pool_size = 7

	roiPoolingResult = roiPooling(pool_size,roiNum)([baseLayer, roiInput])

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

def trainModel():
	imgList = dataProcessor.readJson()
	random.seed(1)
	random.shuffle(imgList)
	trainData = dataProcessor.getData(imgList)

	img_input = Input(shape=(None, None, 3))
	roi_input = Input(shape=(None, 4))

	anchorNum = 9
	classNum = 7

	vggLayer = vgg16(input_tensor = img_input, trainable = True)
	rpn = rpnLayer(vggLayer)
	classifier = classifierLayer(vggLayer,roi_input)

	model_rpn = Model(img_input, rpn[:2])
	model_classifier = Model([img_input,roi_input], classifier)
	model_all = Model([img_input,roi_input], rpn[:2] + classifier)

	optimizer = Adam(lr = 1e-5)
	optimizer_classifier = Adam(lr = 1e-5)
	model_rpn.compile(optimizer = optimizer, loss = [rpn_loss_cls(anchorNum), rpn_loss_regr(anchorNum)])
	model_classifier.compile(optimizer = optimizer_classifier, loss = [class_loss_cls, class_loss_regr(classNum - 1)], metrics={'dense_class_{}'.format(classNum): 'accuracy'})
	model_all.compile(optimizer = 'sgd', loss = 'mae')

	modelPath = '/content/drive/My Drive/Colab Notebooks/model/'
	record_path = modelPath + 'record.csv'

	if not os.path.isfile(modelPath + 'frcnn_vgg.hdf5'):
		try:
			model_rpn.load_weights(modelPath + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
			model_classifier.load_weights(modelPath + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)
		except:
			print('load weights failed')
		record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
	else:
		model_rpn.load_weights(modelPath + 'frcnn_vgg.hdf5', by_name=True)
		model_classifier.load_weights(modelPath + 'frcnn_vgg.hdf5', by_name=True)

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

	num_epochs = 40
	epoch_length = 2000
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
						model_all.save_weights(modelPath + 'frcnn_vgg.hdf5')

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

def testModel():
	pass
