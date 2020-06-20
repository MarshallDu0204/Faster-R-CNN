from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed

from keras.models import Model
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers

import numpy as np
import copy


def readJson(basePath = '/content/drive/My Drive/Colab Notebooks/'):
	with open(basePath+"dataInfo.json","r") as file:
		imgInfo = json.load(fp = file)
		return imgInfo

def augmentImg(singleImg,useAugment = True):

	imgData = copy.deepcopy(singleImg)
	img = cv2.imread(imgData['path'])

	if useAugment:
		rows,cols = img.shape[:2]

		if np.random.randint(0,2)==0:
			img = cv2.flip(img,1)
			bboxesList = imgData['bboxes']
			for line in imgData['bboxes']:
				line[1] = cols-line[1]
				line[2] = cols-line[2]

		if np.random.randint(0,2)==0:
			img = cv2.flip(img,0)
			bboxesList = imgData['bboxes']
			for line in imgData['bboxes']:
				line[3] = rows-line[3]
				line[4] = rows-line[4]

		rotAngle = np.random.choice([0,90,180,270],1)[0]

		if rotAngle == 0:
			pass

		if rotAngle == 90:
			img = np.transpose(img, (1,0,2))
			img = cv2.flip(img,1)
			for line in imgData['bboxes']:
				tempX1 = line[1]
				tempX2 = line[2]
				line[1] = rows-line[4]
				line[2] = rows-line[3]
				line[3] = tempX1
				line[4] = tempX2

		if rotAngle == 180:
			img = cv2.flip(img,-1)
			for line in imgData['bboxes']:
				tempX = line[2]
				tempY = line[4]
				line[2] = cols-line[1]
				line[1] = cols-tempX
				line[4] = rows-line[3]
				line[3] = rows-tempY

		if rotAngle == 270:
			img = np.transpose(img, (1,0,2))
			img = cv2.flip(img,0)
			for line in imgData['bboxes']:
				tempX1 = line[1]
				tempX2 = line[2]
				line[1] = line[3]
				line[2] = line[4]
				line[3] = cols-tempX2
				line[4] = cols-tempX1

	imgData['width'] = img.shape[1]
	imgData['height'] = img.shape[0]

	return imgData,img

def resizeImg(oriWidth,oriHeight,limit = 300):
	if oriWidth<= oriHeight:
		div = float(limit)/oriWidth
		outWidth = limit
		outHeight = int(oriHeight*div)
	else:
		div = float(limit)/oriHeight
		outWidth = int(oriWidth*div)
		outHeight = limit

	return outWidth,outHeight

def calcIoU():
	pass

def calcAnchors(imgData,width,height,resizeWidth,resizeHeight,shrinkFactor = 16):
	pass

def getData(imgInfo,basePath = '/content/drive/My Drive/Colab Notebooks/',useAugment = True):

	for imgData in imgInfo:
		try:
			if useAugment:
				imgData,img = augmentImg(imgData)
			else:
				imgData,img = augmentImg(imgData,useAugment = False)
			resizeWidth,resizeHeight = resizeImg(imgData['width'],imgData['height'])
			img = cv2.resize(img,(resizeWidth,resizeHeight),interpolation=cv2.INTER_CUB)

			biClass,regr = calcAnchors(imgData,imgData['width'],imgData['height'],resizeWidth,resizeHeight)

			#img = img[:,:, (2, 1, 0)] #BGR--> RGB
			img = img.astype(np.float32)
			channelMean = [103.939, 116.779, 123.68]
			#channelMean = [114.45,105.69,98.96]
			img[:,:,0] = img[:,:,0]-channelMean[0]
			img[:,:,1] = img[:,:,1]-channelMean[1]
			img[:,:,2] = img[:,:,2]-channelMean[2]

			img = np.transpose(img,(2,0,1))
			img = np.expand_dims(x_img, axis=0)
			img = np.transpose(img, (0, 2, 3, 1))

		except Exception as e:
			print(e)
			continue

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

def rpn(baseLayer):

	anchorNum = 9

	x = Conv2D(512,(3,3),activation = 'relu',padding= 'same',kernel_initializer='normal')(baseLayer)

	biclassX = Conv2D(anchorNum,(1,1),activation = 'sigmoid',kernel_initializer='uniform')(x)

	regrX = Conv2D(anchorNum*4,(1,1),activation = 'linear',kernel_initializer='zero')(x)

	return [biclassX,regrX,baseLayer]

