import pandas as pd
from tqdm import tqdm
import os
import csv
from urllib.request import urlretrieve
import threading
import math
import json
import cv2
import shutil
import numpy as np

outputInfo = []
downloadedSet = set()
downloadedCont = 0

def getDataLength(path = 'oidv6-train-annotations-bbox.csv'):
	count = 0
	fp = open(path, 'r', encoding='utf-8')
	while 1:
		buffer = fp.read(8 * 1024 * 1024)
		if not buffer:
			break
		count += buffer.count('\n')

	fp.close()
	return count

def getImgDict():
	imgDict = {'/m/01g317':'Person', '/m/01c648':'Laptop',
	'/m/01yrx':'Cat', '/m/06nrc':'Shotgun', '/m/0k4j':'Car', '/m/0dzct':'Human face'}
	return imgDict

def getFullClass():
	fullClass = {'Person':1, 'Laptop':2, 'Cat':3, 'Shotgun':4, 'Car':5, 'Human face':6, 'BG':7}
	return fullClass
	
def readData(chunkLen = 300000):
	infoSet = {'/m/01g317', '/m/01c648', '/m/01yrx', '/m/06nrc', '/m/0k4j', '/m/0dzct'}
	infoList = []
	
	chunks = pd.read_csv('oidv6-train-annotations-bbox.csv', chunksize = chunkLen)
	i = 0
	for chunk in chunks:
		for index in tqdm(range(chunkLen)):
			if chunk.loc[index][chunk.index[2]] in infoSet:
				tempList = []
				tempList.append(chunk.loc[index][chunk.index[0]])
				tempList.append(chunk.loc[index][chunk.index[2]])
				tempList.append(chunk.loc[index][chunk.index[4]])
				tempList.append(chunk.loc[index][chunk.index[5]])
				tempList.append(chunk.loc[index][chunk.index[6]])
				tempList.append(chunk.loc[index][chunk.index[7]])
				infoList.append(tempList)
				
		i += 1
		if i == 1:#only download the first chunk is enough!
			break

	return infoList

def slice(arr, m):
	n = int(math.ceil(len(arr) / float(m)))
	return [arr[i:i + n] for i in range(0, len(arr), n)]


def loadImgSingle(subInfoList, nameList, urlList):
	global downloadedSet
	global outputInfo
	global downloadedCont
	initDir = "D://"
	imgDict = getImgDict()
	for item in tqdm(subInfoList):
		downloadDir = initDir + 'data/' + imgDict[item[1]]
		img_name = item[0] + ".jpg"
		if img_name not in downloadedSet:
			downloadedSet.add(img_name)
			url = urlList[nameList.index(img_name)]
			saveDir = downloadDir + "/" + img_name
			urlretrieve(url,saveDir)
			downloadedCont += 1
		tempList = [item[0],imgDict[item[1]],item[2],item[3],item[4],item[5]]
		outputInfo.append(tempList)

def downloadImgThread(infoList, threadNum = 10, initDir = 'D://'):
	global downloadedSet
	global outputInfo
	global downloadedCont

	if not os.path.exists(initDir + "data"):
		os.mkdir(initDir + "data")
	
	imgDict = getImgDict()

	for key in imgDict:
		if not os.path.exists(initDir + "data/" + imgDict[key]):
			os.mkdir(initDir + "data/" + imgDict[key])

	nameList = []
	urlList = []
	infoData = []

	with open("train-images-boxable.csv", 'r', encoding = 'utf-8') as file:
		lines = csv.reader(file)
		for line in lines:
			infoData.append(line)
		infoData = infoData[1:len(infoData)]
		for element in infoData:
			nameList.append(element[0])
			urlList.append(element[1])

	sliceList = slice(infoList,threadNum)
	
	threadList = []
	for i in range(threadNum):
		threadList.append(threading.Thread(target = loadImgSingle, args = (sliceList[i],nameList,urlList)))

	for i in range(threadNum):
		threadList[i].start()

	for i in range(threadNum):
		threadList[i].join()

	with open('imgInfo.txt','w',encoding = 'utf-8') as file:
		for element in tqdm(outputInfo):
			tempStr = str(element[0]) + "|" + str(element[1]) + "|" + str(element[2]) + "|" + str(element[3]) + "|" + str(element[4]) + "|" + str(element[5]) + "\n"
			file.writelines(tempStr)

def integrateImg(basePath = 'D://'):
	dirList = os.listdir(basePath + 'data/')
	for item in tqdm(dirList):
		dirPath = basePath + 'data/' + item + '/'
		imgList = os.listdir(dirPath)
		for img in imgList:
			srcPath = dirPath + img
			destPath = basePath + 'data/' + img
			shutil.move(srcPath,destPath)
		os.rmdir(dirPath)


def retriveData(basePath = 'D://'):
	chartDict = {'Person':1, 'Laptop':2, 'Cat':3, 'Shotgun':4, 'Car':5, 'Human face':6}

	imgInfo = {}
	imgInfoList = "imgInfo.txt"
	colabPath = '/content/drive/My Drive/Colab Notebooks/data/'
	imgDir = basePath+"data/"

	with open(imgInfoList, 'r', encoding = 'utf-8') as file:
		infoList = file.readlines()
		
		for element in tqdm(infoList):
			tempList = []
			element = element.split("|")
			element[5] = element[5].strip()
			if element[0] not in imgInfo:
				img = cv2.imread(imgDir + element[0] + ".jpg")
				(rows,cols) = img.shape[:2]
				tempDict = {}
				tempDict['path'] = colabPath + element[0] + ".jpg"
				tempDict['width'] = cols
				tempDict['height'] = rows
				tempDict['bboxes'] = []
				imgInfo[element[0]] = tempDict
			imgWidth = imgInfo[element[0]]['width']
			imgHeight = imgInfo[element[0]]['height']
			classNum = int(chartDict[element[1]])
			xMin = int(imgWidth*float(element[2]))
			xMax = int(imgWidth*float(element[3]))
			yMin = int(imgHeight*float(element[4]))
			yMax = int(imgHeight*float(element[5]))
			boxInfo = [classNum, xMin, xMax, yMin, yMax]
			imgInfo[element[0]]['bboxes'].append(boxInfo)
			
	return imgInfo

def saveJson(imgInfo):
	with open("dataInfo.json", "w") as file:
		json.dump(imgInfo, file, ensure_ascii=False)

def computeChannelMean(basePath = 'D://'):
	imgDir = basePath + 'data/'
	imgList = os.listdir(imgDir)
	firstChannel = []
	secondChannel = []
	thirdChannel = []
	
	for item in tqdm(imgList):
		imgPath = imgDir + item
		img = cv2.imread(imgPath)
		img = img[:,:, (2, 1, 0)]
		img = img.astype(np.float32)
		firstChannel.append(np.mean(img[:,:,0]))
		secondChannel.append(np.mean(img[:,:,1]))
		thirdChannel.append(np.mean(img[:,:,2]))

	firstChannel = np.array(firstChannel)
	secondChannel = np.array(secondChannel)
	thirdChannel = np.array(thirdChannel)
	avg1 = np.mean(firstChannel)
	avg2 = np.mean(secondChannel)
	avg3 = np.mean(thirdChannel)

	print(avg1,avg2,avg3)
	#114.45 105.69 98.96
'''
infoList = readData()
downloadImgThread(infoList,threadNum = 20)
'''
#integrateImg()
'''
imgInfo = retriveData()
saveJson(imgInfo)
'''
#computeChannelMean()

