import scipy.misc
import numpy as np
import os
import struct
import random
import copy
import math
from random import shuffle

#The IAM dataset is converted into one complete file and a split between test and training exmaples is performed.
#This improves the speed of data loading.
#The resulting data files have the following structure (Everything is stored as binary data):
#   (int) num Elements in Sequence
#   (ascii*numElements) Contains the sequence and converts it into ascii
#   (int) image height
#   (int) image width
#   (float *width * height) image data
#This is then followed by the same structure for the next data point
#Additionally it returns the meta file which contains the sequence, max image dimensions and max sequence length.
def ConvertToFile(loadPath, labelFile, newFileTrain, newFileTest, newFileVal, metaFileName, useAll):
    lFile = open(labelFile, 'r')

    outputFile = [open(newFileTest, 'wb'), open(newFileVal, 'wb'), open(newFileTrain, 'wb')]

    metaFile = open(metaFileName, 'wb')

    fileNames = []
    okay = []
    labels = []
    totalString = ""

    for line in lFile:
        sLine = line.split(sep=' ')
        fileNames.append(sLine[0])
        okay.append((sLine[1] == 'ok'))

        if sLine[-1].endswith('\n'):
            newString = sLine[-1][:-1]

        else:
            newString = sLine[-1]

        labels.append(newString)
        if okay[-1] or useAll:
            totalString = totalString + newString

    fileNamesTmp = copy.deepcopy(fileNames)

    random.shuffle(fileNamesTmp)

    lenNames = len(fileNamesTmp)
    endTest = math.floor(lenNames *0.2*0.8)
    endValid = math.floor(lenNames *0.2)
    test_data = fileNamesTmp[:endTest]
    valid_data = fileNamesTmp[endTest:endValid]
    train_data = fileNamesTmp[endValid:lenNames]

    useCase = {**{k:'test' for k in test_data}, **{k:'valid' for k in valid_data}, **{k:'train' for k in train_data}}
    translate = {'test':0, 'valid':1, 'train': 2}

    okayDict = dict(zip(fileNames, okay))
    labelsDict = dict(zip(fileNames, labels))
    allChars = list(set(totalString))

    metaFile.write(struct.pack('i', len(allChars)))
    for i in allChars:
        metaFile.write(str.encode(i))

    maxWidth = 0
    maxHeight = 0
    maxSeqLen = 0

    all_names = list()
    for path, subdirs, files in os.walk(loadPath):
        for name in files:
            all_names.append([name,path])
    shuffle(all_names)
    for name,path in all_names:
        tmpName = name[:-4]
        if okayDict[tmpName] or (useAll): #and useCase[tmpName] == 2):
            localLabel = labelsDict[tmpName]
            writeFile = outputFile[translate[useCase[tmpName]]]
            if maxSeqLen < len(localLabel) + 1:
                maxSeqLen = len(localLabel) + 1
            writeFile.write(struct.pack('i', len(localLabel)))
            writeFile.write(str.encode(localLabel))
            img = scipy.misc.imread(os.path.join(path, name))
            imgShape = img.shape

            if imgShape[0] > maxHeight:
                maxHeight = imgShape[0]
            if imgShape[1] > maxWidth:
                maxWidth = imgShape[1]
            writeFile.write(struct.pack('i', imgShape[0]))
            writeFile.write(struct.pack('i', imgShape[1]))
            img = 1 - (img / 255)
            img.tofile(writeFile)

    metaFile.write(struct.pack('i', maxHeight))
    metaFile.write(struct.pack('i', maxWidth))
    metaFile.write(struct.pack('i', maxSeqLen))

#used to read the meta data file.
def ReadMetaData(metaFile):
    lenChar = struct.unpack('i', metaFile.read(4))[0]
    chars = metaFile.read(lenChar).decode('ascii')
    tmp = struct.unpack('iii', metaFile.read(12))
    return chars, tmp[0], tmp[1], tmp[2]

#Reads the next image and the contained sequence from dataFile.
def ReadNextEntry(dataFile):
    #Read the current length of the sequence
    byt = dataFile.read(4)

    #Restart when at the end of file
    if len(byt) < 4:
        print("NextIteration")
        dataFile.seek(0, 0)
        byt = dataFile.read(4)
    lenLocalLabel = struct.unpack('i', byt)[0]

    #read the character sequence (servers as label)
    label = dataFile.read(lenLocalLabel).decode('ascii')
    #get image dimensions
    tmp = struct.unpack('ii', dataFile.read(8))
    height = tmp[0]
    width = tmp[1]
    #read image
    data = np.fromfile(dataFile, count=height*width)
    data = np.reshape(data, (height, width))
    return label, data

#The same as ReadNextEntry but end when reaching the end of the file.
def ReadNextEntryEnd(dataFile):
    byt = dataFile.read(4)
    if len(byt) < 4:
        return None, None
    lenLocalLabel = struct.unpack('i', byt)[0]
    label = dataFile.read(lenLocalLabel).decode('ascii')
    tmp = struct.unpack('ii', dataFile.read(8))
    height = tmp[0]
    width = tmp[1]
    data = np.fromfile(dataFile, count=height*width)
    data = np.reshape(data, (height, width))
    return label, data

def Reset(dataFile):
    dataFile.seek(0,0)

def decode(indices, values):
    strings = list()
    string = list()
    current = 0
    for idx, i in enumerate(indices):
        if i[0] == current:
            string.append(values[idx])
        else:
            current = current +1
            strings.append((string))
            string = list()
            string.append(values[idx])

    return strings

#compute the word error rate
def wer(r, h):
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    if len(r) != 0:
        return d[len(r)][len(h)]/len(r)
    else:
        return 0

# ConvertToFile('D:\MLProjects\DataSets\IAM\words', 'D:\MLProjects\DataSets\IAM\IAMLabels.txt', 'trainData.dat', 'testData.dat', 'validData.dat', 'metaFile.dat', False)

def CountElements(file_name):
    file = open(file_name, 'rb')
    counter = 0
    while True:
        res1, res2 = ReadNextEntryEnd(file)
        if res1 is None and res2 is None:
            return counter
        counter += 1

#print("Amount Valid: " + str(CountElements('validData.dat')))
#print("Amount Test: " + str(CountElements('testData.dat')))
#print("Amount Train: " + str(CountElements('trainData.dat')))

