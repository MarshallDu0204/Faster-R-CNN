import numpy as np
import copy
import math
import random
import json
import cv2
import dataGenerator

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

		rotAngle = np.random.choice([0, 90, 180, 270], 1)[0]

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
		
	xLeft = max(regionA[0], regionB[0])
	xRight = min(regionA[2], regionB[2])
	yTop = max(regionA[1], regionB[1])
	yBottom = min(regionA[3], regionB[3])

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
	shrinkFactor = float(shrinkFactor)

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
				curX1 = shrinkFactor * (pixelX + 0.5) - anchorX / 2.0
				curX2 = shrinkFactor * (pixelX + 0.5) + anchorX / 2.0
				if curX1 < 0 or curX2 > resizeWidth:
					continue

				for pixelY in range(outHeight):
					curY1 = shrinkFactor * (pixelY + 0.5) - anchorY / 2.0
					curY2 = shrinkFactor * (pixelY + 0.5) + anchorY / 2.0
					if curY1 < 0 or curY2 > resizeHeight:
						continue
					bboxesType = 'false'
					bestPosIoU = 0.0

					for item in range(boxNum):
						regionA = [groudTruthAnchor[item,0], groudTruthAnchor[item,2], groudTruthAnchor[item,1], groudTruthAnchor[item,3]]
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
			anc_valid[bestAnchor[index,0], bestAnchor[index,1], bestAnchor[index,2] + 3 * bestAnchor[index,3]] = 1
			anc_obj[bestAnchor[index,0], bestAnchor[index,1], bestAnchor[index,2] + 3 * bestAnchor[index,3]] = 1
			regrIndex = 4 * (bestAnchor[index,2] + 3 * bestAnchor[index,3])
			anc_regr[bestAnchor[index,0], bestAnchor[index,1], regrIndex:regrIndex + 4] = bestBoxRegr[index,:]
	
	anc_obj = np.transpose(anc_obj, (2,0,1))
	anc_obj = np.expand_dims(anc_obj, axis = 0)
	anc_valid = np.transpose(anc_valid, (2,0,1))
	anc_valid = np.expand_dims(anc_valid, axis = 0)
	anc_regr = np.transpose(anc_regr, (2,0,1))
	anc_regr = np.expand_dims(anc_regr,axis = 0)

	pos_anchor = np.where(np.logical_and(anc_obj[0, :, :, :] == 1, anc_valid[0, :, :, :] == 1))
	neg_anchor = np.where(np.logical_and(anc_obj[0, :, :, :] == 0, anc_valid[0, :, :, :] == 1))

	posAnchorNum = len(pos_anchor[0])

	limitAnchorNum = 256
	
	if len(pos_anchor[0]) > limitAnchorNum / 2:
		ignoreAnchor = random.sample(range(len(pos_anchor[0])), len(pos_anchor[0]) - limitAnchorNum / 2)
		anc_valid[0, pos_anchor[0][ignoreAnchor], pos_anchor[1][ignoreAnchor], pos_anchor[2][ignoreAnchor]] = 0
		posAnchorNum = limitAnchorNum/2

	if len(neg_anchor[0]) + posAnchorNum > limitAnchorNum:
		ignoreAnchor = random.sample(range(len(neg_anchor[0])), len(neg_anchor[0]) - posAnchorNum)
		anc_valid[0, neg_anchor[0][ignoreAnchor], neg_anchor[1][ignoreAnchor], neg_anchor[2][ignoreAnchor]] = 0
	
	rpn_Class = np.concatenate([anc_valid, anc_obj], axis = 1)
	rpn_Regr = np.concatenate([np.repeat(anc_obj, 4, axis = 1), anc_regr], axis = 1)
	return np.copy(rpn_Class),np.copy(rpn_Regr)

def applyRegr(anchorMatrix,regrInfo):
	try:
		x = anchorMatrix[0, :, :]
		y = anchorMatrix[1, :, :]
		w = anchorMatrix[2, :, :]
		h = anchorMatrix[3, :, :]

		tx = regrInfo[0, :, :]
		ty = regrInfo[1, :, :]
		tw = regrInfo[2, :, :]
		th = regrInfo[3, :, :]

		centerX = x + w / 2.
		centerY = y + h / 2.
		centerX1 = tx * w + centerX
		centerY1 = ty * h + centerY

		w1 = np.exp(tw.astype(np.float64)) * w
		h1 = np.exp(th.astype(np.float64)) * h
		x1 = centerX1 - w1 / 2.
		y1 = centerY1 - h1 / 2.

		x1 = np.round(x1)
		y1 = np.round(y1)
		w1 = np.round(w1)
		h1 = np.round(h1)

		return np.stack([x1,y1,w1,h1])

	except Exception as e:
		print(e)
		return anchorMatrix

def non_max_suppression_fast(boxes, probs, overlap_threshold = 0.7, max_boxes = 300):
	if len(boxes) == 0:
		return []

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []
	area = (x2 - x1) * (y2 - y1)
	idxs = np.argsort(probs)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		ww_int = np.maximum(0, xx2_int - xx1_int)
		hh_int = np.maximum(0, yy2_int - yy1_int)

		area_int = ww_int * hh_int
		area_union = area[i] + area[idxs[:last]] - area_int
		overlap = area_int/(area_union + 1e-6)

		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_threshold)[0])))

		if len(pick) >= max_boxes:
			break

	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes

def proposalCreator(rpn_class, rpn_regr, max_boxes = 300, overlap_threshold = 0.7):
	shrinkFactor = 16.0
	shrinkFactor = float(shrinkFactor)
	stdevScaleFactor = 4.0

	rpn_regr = rpn_regr / stdevScaleFactor
	anchorSize = [64,128,256]
	anchorRatio = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]
	
	(rows,cols) = rpn_class.shape[1:3]
	curLayer = 0

	anchorMatrix = np.zeros((4, rpn_class.shape[1], rpn_class.shape[2], rpn_class.shape[3]))

	for size in range(len(anchorSize)):
		for ratio in range(len(anchorRatio)):
			anchorX = (anchorSize[size] * anchorRatio[ratio][0]) / shrinkFactor
			anchorY = (anchorSize[size] * anchorRatio[ratio][1]) / shrinkFactor

			regrInfo = rpn_regr[0, :, :, 4 * curLayer:4 * curLayer + 4]
			regrInfo = np.transpose(regrInfo, (2, 0, 1))

			X,Y = np.meshgrid(np.arange(cols),np.arange(rows))

			anchorMatrix[0, :, :, curLayer] = X - anchorX / 2
			anchorMatrix[1, :, :, curLayer] = Y - anchorY / 2
			anchorMatrix[2, :, :, curLayer] = anchorX
			anchorMatrix[3, :, :, curLayer] = anchorY

			anchorMatrix[:, :, :, curLayer] = applyRegr(anchorMatrix[:, :, :, curLayer], regrInfo)

			anchorMatrix[2, :, :, curLayer] = np.maximum(1, anchorMatrix[2, :, :, curLayer])
			anchorMatrix[3, :, :, curLayer] = np.maximum(1, anchorMatrix[3, :, :, curLayer])

			anchorMatrix[2, :, :, curLayer] += anchorMatrix[0, :, :, curLayer]
			anchorMatrix[3, :, :, curLayer] += anchorMatrix[1, :, :, curLayer]

			anchorMatrix[0, :, :, curLayer] = np.maximum(0, anchorMatrix[0, :, :, curLayer])
			anchorMatrix[1, :, :, curLayer] = np.maximum(0, anchorMatrix[1, :, :, curLayer])
			anchorMatrix[2, :, :, curLayer] = np.minimum(cols - 1, anchorMatrix[2, :, :, curLayer])
			anchorMatrix[3, :, :, curLayer] = np.minimum(rows - 1, anchorMatrix[3, :, :, curLayer])

			curLayer += 1

	allAnchor = np.reshape(anchorMatrix.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
	allProb = rpn_class.transpose((0, 3, 1, 2)).reshape((-1))

	illegalAnchor = np.where((allAnchor[:, 0] - allAnchor[:, 2] >= 0) | (allAnchor[:, 1] - allAnchor[:, 3] >= 0))

	allAnchor = np.delete(allAnchor, illegalAnchor, 0)
	allProb = np.delete(allProb, illegalAnchor, 0)

	result = non_max_suppression_fast(allAnchor, allProb, overlap_threshold = overlap_threshold, max_boxes = max_boxes)

	return result

def roiHead(anchorMatrix, imgData):
	shrinkFactor = 16.0
	shrinkFactor = float(shrinkFactor)
	width = imgData['width']
	height = imgData['height']
	resizeWidth,resizeHeight = resizeImg(width,height)

	boxNum = len(imgData['bboxes'])

	groudTruthAnchor = np.zeros((boxNum, 4))
	
	index = 0
	for item in imgData['bboxes']:
		groudTruthAnchor[index,0] = int(round(item[1] * (resizeWidth / float(width)) / shrinkFactor))
		groudTruthAnchor[index,1] = int(round(item[2] * (resizeWidth / float(width)) / shrinkFactor))
		groudTruthAnchor[index,2] = int(round(item[3] * (resizeHeight / float(height)) / shrinkFactor))
		groudTruthAnchor[index,3] = int(round(item[4] * (resizeHeight / float(height)) / shrinkFactor))
		index += 1

	roiList = []
	classInfo = []
	regrLabel = []
	regrCoord = []

	for index in range(anchorMatrix.shape[0]):
		(x1,y1,x2,y2) = anchorMatrix[index, :]
		x1 = int(round(x1))
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		bestPosIoU = 0.0
		best_bbox = -1
		for item in range(len(imgData['bboxes'])):
			regionA = [groudTruthAnchor[item,0], groudTruthAnchor[item,2], groudTruthAnchor[item,1], groudTruthAnchor[item,3]]
			regionB = [x1, y1, x2, y2]
			curIoU = calcIoU(regionA,regionB)

			if curIoU > bestPosIoU:
				bestPosIoU = curIoU
				best_bbox = item

		if bestPosIoU < 0.1:
			continue
		else:
			w = x2 - x1
			h = y2 - y1
			roiList.append([x1,y1,w,h])

			if 0.1 <= bestPosIoU < 0.5:
				className = 'BG'
				classNum = 7
			elif 0.5 <= bestPosIoU:
				classNum = imgData['bboxes'][best_bbox][0]

				groudTruthCenterX = (groudTruthAnchor[best_bbox,0] + groudTruthAnchor[best_bbox,1]) / 2.0
				groudTruthCenterY = (groudTruthAnchor[best_bbox,2] + groudTruthAnchor[best_bbox,3]) / 2.0
				anchorCenterX = x1 + w / 2.0
				anchorCenterY = y1 + h / 2.0

				tx = (groudTruthCenterX - anchorCenterX) / float(w)
				ty = (groudTruthCenterY - anchorCenterY) / float(h)
				tw = np.log((groudTruthAnchor[best_bbox,1] - groudTruthAnchor[best_bbox,0]) / float(w))
				th = np.log((groudTruthAnchor[best_bbox,3] - groudTruthAnchor[best_bbox,2]) / float(h))
			else:
				raise RuntimeError

		classDict = dataGenerator.getFullClass()
		classLabel = len(classDict) * [0]
		classLabel[classNum - 1] = 1
		classInfo.append(copy.deepcopy(classLabel))
		coords = [0] * 4 * (len(classDict) - 1)
		labels = [0] * 4 * (len(classDict) - 1)
		if classNum != 7:
			classNum = classNum - 1
			pos = 4 * classNum
			regrStd = [8.0, 8.0, 4.0, 4.0]
			coords[pos:pos + 4] = [tx * regrStd[0], ty * regrStd[1], tw * regrStd[2], th * regrStd[3]]
			labels[pos:pos + 4] = [1,1,1,1]
			regrCoord.append(copy.deepcopy(coords))
			regrLabel.append(copy.deepcopy(labels))
		else:
			regrCoord.append(copy.deepcopy(coords))
			regrLabel.append(copy.deepcopy(labels))

	if len(roiList) == 0:
		return None,None,None

	X2 = np.array(roiList)
	Y1 = np.array(classInfo)
	Y2 = np.concatenate([np.array(regrLabel),np.array(regrCoord)],axis=1)
	return np.expand_dims(X2,axis = 0), np.expand_dims(Y1,axis = 0), np.expand_dims(Y2,axis = 0)

def roiSelect(Y1,roiNum = 4):
	
	negSamples = np.where(Y1[0, :, -1] == 1)
	posSamples = np.where(Y1[0, :, -1] == 0)

	if len(negSamples) > 0:
		negSamples = negSamples[0]
	else:
		negSamples = []
	if len(posSamples) > 0:
		posSamples = posSamples[0]
	else:
		posSamples = []

	posNum = len(posSamples)

	if len(posSamples) < roiNum // 2:
		selectPos = posSamples.tolist()
	else:
		selectPos = np.random.choice(posSamples, roiNum // 2, replace=False).tolist()

	try:
		selectNeg = np.random.choice(negSamples, roiNum - len(selectPos), replace=False).tolist()
	except:
		selectNeg = np.random.choice(negSamples, roiNum - len(selectPos), replace=True).tolist()

	selectSamples = selectPos + selectNeg

	return copy.deepcopy(selectSamples), posNum


def getData(imgInfo,basePath = '/content/drive/My Drive/Colab Notebooks/',useAugment = True):

	while True:

		for imgData in imgInfo:
			try:
				if useAugment:
					imgData,img = augmentImg(imgData)
				else:
					imgData,img = augmentImg(imgData,useAugment = False)
				resizeWidth,resizeHeight = resizeImg(imgData['width'],imgData['height'])
				img = cv2.resize(img, (resizeWidth,resizeHeight), interpolation=cv2.cv2.INTER_CUBIC)
				try:
					rpn_Class,rpn_Regr = calcAnchors(imgData, imgData['width'], imgData['height'], resizeWidth, resizeHeight)
				except:
					continue

				img = img.astype(np.float32)
				channelMean = [103.939, 116.779, 123.68]
				img[:,:,0] = img[:,:,0] - channelMean[0]
				img[:,:,1] = img[:,:,1] - channelMean[1]
				img[:,:,2] = img[:,:,2] - channelMean[2]

				img = np.transpose(img, (2,0,1))
				img = np.expand_dims(img, axis=0)
				img = np.transpose(img, (0, 2, 3, 1))

				stdevScaleFactor = 4.0
				rpn_Regr[:,rpn_Regr.shape[1]//2:, :, :] *= stdevScaleFactor

				rpn_Class = np.transpose(rpn_Class, (0, 2, 3, 1))
				rpn_Regr = np.transpose(rpn_Regr, (0, 2, 3, 1))

				yield np.copy(img),[np.copy(rpn_Class),np.copy(rpn_Regr)],imgData

			except Exception as e:
				print(e)
				continue