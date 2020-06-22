import numpy as np
import copy
import math
import random
import json
import cv2

def readJson(basePath = '/content/drive/My Drive/Colab Notebooks/'):
	with open(basePath+"dataInfo.json","r") as file:
		imgInfo = json.load(fp = file)
		imgList = []
		for key in imgInfo:
			imgList.append(imgInfo[key])
		return imgList

def augmentImg(singleImg,useAugment = True):
	imgData = copy.deepcopy(singleImg)
	img = cv2.imread(imgData['path'])

	if useAugment:
		rows,cols = img.shape[:2]

		if np.random.randint(0,2) == 0:
			img = cv2.flip(img,1)
			bboxesList = imgData['bboxes']
			for line in imgData['bboxes']:
				tempX = line[1]
				line[1] = cols - line[2]
				line[2] = cols - tempX

		if np.random.randint(0,2) == 0:
			img = cv2.flip(img,0)
			bboxesList = imgData['bboxes']
			for line in imgData['bboxes']:
				tempY = line[3]
				line[3] = rows - line[4]
				line[4] = rows - tempY

		rotAngle = np.random.choice([0,90,180,270],1)[0]

		if rotAngle == 0:
			pass

		if rotAngle == 90:
			img = np.transpose(img, (1,0,2))
			img = cv2.flip(img,1)
			for line in imgData['bboxes']:
				tempX1 = line[1]
				tempX2 = line[2]
				line[1] = rows - line[4]
				line[2] = rows - line[3]
				line[3] = tempX1
				line[4] = tempX2

		if rotAngle == 180:
			img = cv2.flip(img,-1)
			for line in imgData['bboxes']:
				tempX = line[2]
				tempY = line[4]
				line[2] = cols - line[1]
				line[1] = cols - tempX
				line[4] = rows - line[3]
				line[3] = rows - tempY

		if rotAngle == 270:
			img = np.transpose(img, (1,0,2))
			img = cv2.flip(img,0)
			for line in imgData['bboxes']:
				tempX1 = line[1]
				tempX2 = line[2]
				line[1] = line[3]
				line[2] = line[4]
				line[3] = cols - tempX2
				line[4] = cols - tempX1

	imgData['width'] = img.shape[1]
	imgData['height'] = img.shape[0]

	return imgData,img

def resizeImg(oriWidth,oriHeight,limit = 300):
	if oriWidth <= oriHeight:
		div = float(limit) / oriWidth
		outWidth = limit
		outHeight = int(oriHeight * div)
	else:
		div = float(limit) / oriHeight
		outWidth = int(oriWidth * div)
		outHeight = limit

	return outWidth,outHeight

def calcIoU(regionA,regionB):
	if regionA[0] > regionA[2] or regionA[1] > regionA[3] or regionB[0] > regionB[2] or regionB[1] > regionB[3]:
		return 0.0
		
	xLeft = max(regionA[0],regionB[0])
	xRight = min(regionA[2],regionB[2])
	yTop = max(regionA[1],regionB[1])
	yBottom = min(regionA[3],regionB[3])

	if xRight < xLeft or yBottom < yTop:
		return 0.0

	intersec = (xRight - xLeft) * (yBottom - yTop)

	areaA = (regionA[2] - regionA[0]) * (regionA[3] - regionA[1])
	areaB = (regionB[2] - regionB[0]) * (regionB[3] - regionB[1])

	union = areaA + areaB - intersec

	return float(intersec) / float(union + 1e-6)
	

def calcAnchors(imgData,width,height,resizeWidth,resizeHeight,shrinkFactor = 16):
	minIoU = 0.3
	maxIoU = 0.7
	anchorNum = 9
	anchorSize = [64,128,256]
	anchorRatio = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
	outWidth = resizeWidth // shrinkFactor
	outHeight = resizeHeight // shrinkFactor

	anc_obj = np.zeros((outHeight, outWidth, anchorNum))
	anc_valid = np.zeros((outHeight, outWidth, anchorNum))
	anc_regr = np.zeros((outHeight, outWidth, anchorNum * 4))

	boxNum = len(imgData['bboxes'])
	
	bboxesAnchorNum = np.zeros(boxNum).astype(int)
	bestAnchor = -1 * np.zeros((boxNum, 4)).astype(int)
	bestIoU = np.zeros(boxNum).astype(np.float32)
	bestBoxRegr = np.zeros((boxNum, 4)).astype(np.float32)

	groudTruthAnchor = np.zeros((boxNum, 4))
	
	index = 0
	for item in imgData['bboxes']:
		groudTruthAnchor[index,0] = item[1] * (resizeWidth / float(width))
		groudTruthAnchor[index,1] = item[2] * (resizeWidth / float(width))
		groudTruthAnchor[index,2] = item[3] * (resizeHeight / float(height))
		groudTruthAnchor[index,3] = item[4] * (resizeHeight / float(height))
		index += 1

	for size in range(len(anchorSize)):

		for ratio in range(len(anchorRatio)):
			anchorX = anchorSize[size] * anchorRatio[ratio][0]
			anchorY = anchorSize[size] * anchorRatio[ratio][1]

			for pixelX in range(outWidth):
				curX1 = shrinkFactor * (pixelX + 0.5) - anchorX / 2
				curX2 = shrinkFactor * (pixelX + 0.5) + anchorX / 2
				if curX1 < 0 or curX2 > resizeWidth:
					continue

				for pixelY in range(outHeight):
					curY1 = shrinkFactor * (pixelY + 0.5) - anchorY / 2
					curY2 = shrinkFactor * (pixelY + 0.5) + anchorY / 2
					if curY1 < 0 or curY2 > resizeHeight:
						continue
					bboxesType = 'false'
					bestPosIoU = 0.0

					for item in range(boxNum):
						regionA = [groudTruthAnchor[item,0],groudTruthAnchor[item,2],groudTruthAnchor[item,1],groudTruthAnchor[item,3]]
						regionB = [curX1,curY1,curX2,curY2]
						anchorIoU = calcIoU(regionA,regionB)

						if anchorIoU > bestIoU[item] or anchorIoU > maxIoU:
							groudTruthCenterX = (groudTruthAnchor[item,0] + groudTruthAnchor[item,1]) / 2.0
							groudTruthCenterY = (groudTruthAnchor[item,2] + groudTruthAnchor[item,3]) / 2.0
							anchorCenterX = (curX1 + curX2) / 2.0
							anchorCenterY = (curY1 + curY2) / 2.0

							tx = (groudTruthCenterX - anchorCenterX) / (curX2 - curX1)
							ty = (groudTruthCenterY - anchorCenterY) / (curY2 - curY1)
							tw = np.log((groudTruthAnchor[item,1] - groudTruthAnchor[item,0]) / (curX2 - curX1))
							th = np.log((groudTruthAnchor[item,3] - groudTruthAnchor[item,2]) / (curY2 - curY1))

						if anchorIoU > bestIoU[item]:
							bestAnchor[item] = [pixelY, pixelX, ratio, size]
							bestIoU[item] = anchorIoU
							bestBoxRegr[item,:] = [tx, ty, tw, th]
						
						if anchorIoU > maxIoU:
							bboxesType = 'true'
							bboxesAnchorNum[item] += 1
							if anchorIoU > bestPosIoU:
								bestPosIoU = anchorIoU
								bestPosRegr = (tx,ty,tw,th)

						if minIoU < anchorIoU < maxIoU:
							if bboxesType != 'true':
								bboxesType = 'neutral'

					if bboxesType == 'false':
						anc_valid[pixelY,pixelX,ratio + 3 * size] = 1
						anc_obj[pixelY,pixelX,ratio + 3 * size] = 0

					elif bboxesType == 'neutral':
						anc_valid[pixelY,pixelX,ratio + 3 * size] = 0
						anc_obj[pixelY,pixelX,ratio + 3 * size] = 0
					
					elif bboxesType == 'true':
						anc_valid[pixelY,pixelX,ratio+3 * size] = 1
						anc_obj[pixelY,pixelX,ratio+3 * size] = 1
						regrIndex = 4 * (ratio + 3 * size)
						anc_regr[pixelY,pixelX,regrIndex:regrIndex + 4] = bestPosRegr

	for index in range(bboxesAnchorNum.shape[0]):
		if bboxesAnchorNum[index] == 0:
			if bestAnchor[index,0] == -1:
				continue
			anc_valid[bestAnchor[index,0],bestAnchor[index,1],bestAnchor[index,2] + 3 * bestAnchor[index,3]] = 1
			anc_obj[bestAnchor[index,0],bestAnchor[index,1],bestAnchor[index,2] + 3 * bestAnchor[index,3]] = 1
			regrIndex = 4 * (bestAnchor[index,2] + 3 * bestAnchor[index,3])
			anc_regr[bestAnchor[index,0],bestAnchor[index,1],regrIndex:regrIndex + 4] = bestBoxRegr[index,:]
	
	anc_obj = np.transpose(anc_obj,(2,0,1))
	anc_obj = np.expand_dims(anc_obj,axis = 0)

	anc_valid = np.transpose(anc_valid,(2,0,1))
	anc_valid = np.expand_dims(anc_valid,axis = 0)

	anc_regr = np.transpose(anc_regr,(2,0,1))
	anc_regr = np.expand_dims(anc_regr,axis = 0)

	
	pos_anchor = np.where(np.logical_and(anc_obj[0, :, :, :] == 1, anc_valid[0, :, :, :] == 1))
	neg_anchor = np.where(np.logical_and(anc_obj[0, :, :, :] == 0, anc_valid[0, :, :, :] == 1))

	posAnchorNum = len(pos_anchor[0])

	limitAnchorNum = 256
	
	if len(pos_anchor[0]) > limitAnchorNum / 2:
		ignoreAnchor = random.sample(range(len(pos_anchor[0])),len(pos_anchor[0]) - limitAnchorNum / 2)
		anc_valid[0,pos_anchor[0][ignoreAnchor],pos_anchor[1][ignoreAnchor],pos_anchor[2][ignoreAnchor]] = 0
		posAnchorNum = limitAnchorNum/2

	if len(neg_anchor[0]) + posAnchorNum > limitAnchorNum:
		ignoreAnchor = random.sample(range(len(neg_anchor[0])),len(neg_anchor[0]) - posAnchorNum)
		anc_valid[0,neg_anchor[0][ignoreAnchor],neg_anchor[1][ignoreAnchor],neg_anchor[2][ignoreAnchor]] = 0
	
	rpn_Class = np.concatenate([anc_valid, anc_obj], axis = 1)
	rpn_Regr = np.concatenate([np.repeat(anc_obj, 4, axis = 1), anc_regr], axis = 1)
	return np.copy(rpn_Class),np.copy(rpn_Regr)

def getData(imgInfo,basePath = '/content/drive/My Drive/Colab Notebooks/',useAugment = True):

	for imgData in imgInfo:
		try:
			if useAugment:
				imgData,img = augmentImg(imgData)
			else:
				imgData,img = augmentImg(imgData,useAugment = False)
			resizeWidth,resizeHeight = resizeImg(imgData['width'],imgData['height'])
			img = cv2.resize(img,(resizeWidth,resizeHeight),interpolation=cv2.cv2.INTER_CUBIC)

			try:
				rpn_Class,rpn_Regr = calcAnchors(imgData,imgData['width'],imgData['height'],resizeWidth,resizeHeight)
			except:
				continue

			#img = img[:,:, (2, 1, 0)] #BGR--> RGB
			img = img.astype(np.float32)
			channelMean = [103.939, 116.779, 123.68]
			#channelMean = [114.45,105.69,98.96]
			img[:,:,0] = img[:,:,0] - channelMean[0]
			img[:,:,1] = img[:,:,1] - channelMean[1]
			img[:,:,2] = img[:,:,2] - channelMean[2]

			img = np.transpose(img,(2,0,1))
			img = np.expand_dims(x_img, axis=0)
			img = np.transpose(img, (0, 2, 3, 1))

			stdevScaleFactor = 4.0

			rpn_Regr[:,rpn_Regr.shape[1]//2:, :, :] *= stdevScaleFactor

			rpn_Class = np.transpose(rpn_Class, (0, 2, 3, 1))
			rpn_Regr = np.transpose(rpn_Regr, (0, 2, 3, 1))

			yield img,[rpn_Class,rpn_Regr],imgData

		except Exception as e:
			print(e)
			continue