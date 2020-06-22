from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed

from keras.models import Model
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers


def vgg16(input_tensor = None, trainable = False):

	inputImg = Input(shape = (None,None,3))

	x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv1')(inputImg)
	x = Conv2D(64,(3,3),activation = 'relu',padding = 'same',name = 'block1_conv2')(x)
	x = MaxPooling2D((2,2),strides = (2,2),name='block1_pool')(x)

	x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv1')(x)
	x = Conv2D(128,(3,3),activation = 'relu',padding = 'same',name = 'block2_conv2')(x)
	x = MaxPooling2D((2,2),strides = (2,2),name='block2_pool')(x)

	x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv1')(x)
	x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv2')(x)
	x = Conv2D(256,(3,3),activation = 'relu',padding = 'same',name = 'block3_conv3')(x)
	x = MaxPooling2D((2,2),strides = (2,2),name='block3_pool')(x)

	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv1')(x)
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv2')(x)
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block4_conv3')(x)
	x = MaxPooling2D((2,2),strides = (2,2),name='block4_pool')(x)

	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv1')(x)
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv2')(x)
	x = Conv2D(512,(3,3),activation = 'relu',padding = 'same',name = 'block5_conv3')(x)
	
	return x

def rpnLayer(baseLayer):

	anchorNum = 9

	x = Conv2D(512,(3,3),activation = 'relu',padding= 'same',kernel_initializer='normal',name = 'rpn_conv1')(baseLayer)

	biclassX = Conv2D(anchorNum,(1,1),activation = 'sigmoid',kernel_initializer='uniform',name = 'rpn_out_class')(x)

	regrX = Conv2D(anchorNum*4,(1,1),activation = 'linear',kernel_initializer='zero',name = 'rpn_out_regress')(x)

	return [biclassX,regrX,baseLayer]

def proposalCreator():
	pass

def roiPooling():
	pass

def classifier():
	pass


'''
baseLayer = vgg16(trainable = True)
rpn = rpnLayer(baseLayer)
'''

