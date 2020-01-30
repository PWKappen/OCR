import tensorflow as tf
import numpy as np
import utility
import threading
import math

#uses a random shcuffle queue
class CustomAsynchBatchLoader(object):
    def __init__(self, metaFileName, dataFileName, batch_size, capacity, min_after_dequeue):
        self.metaFile = open(metaFileName, 'rb')
        self.allChars, self.maxHeight, self.maxWidth, self.maxSeqLen = utility.ReadMetaData(self.metaFile)
        self.allChars += chr(3)
        self.charDict = list(set(self.allChars))
        self.charTrans = dict(zip(self.charDict, np.arange(len(self.charDict))))
        self.charTransInv = dict((v, k) for k, v in self.charTrans.items())
        self.batchSize = batch_size

        self.dataFileName = dataFileName

        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.maxHeight, self.maxWidth])
        self.dataY = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.seqLen = tf.placeholder(dtype=tf.int32, shape=[None])

        self.queue = tf.RandomShuffleQueue(dtypes=[tf.float32, tf.int32, tf.int32],
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)

        self.enqueue_op = self.queue.enqueue([self.dataX, self.dataY, self.seqLen])

    def get_inputs(self):
        outputImages, outputLabels, outputSeqLen = list(self.queue.dequeue())
        return outputImages, outputLabels, outputSeqLen

    def get_queue(self):
        return self.queue

    def data_iterator(self, file):
        while True:
            labelList = list()
            dataList = list()
            seqLen = np.zeros(self.batchSize, dtype=np.int32)
            for i in range(self.batchSize):
                label, data = utility.ReadNextEntry(file)
                label += chr(3)
                seqLen[i] = len(label)
                labelList.append(label)
                dataList.append(data)

            maxLen = np.amax(seqLen)

            tmpMatrix = np.zeros(shape=(self.batchSize, self.maxHeight, self.maxWidth))

            for i in range(self.batchSize):
                shape = dataList[i].shape
                tmpMatrix[i, :shape[0], :shape[1]] = dataList[i]
            resultLabel = np.full(shape=(self.batchSize, maxLen), fill_value=-1)
            for i in range(self.batchSize):
                for j in range(seqLen[i]):
                    resultLabel[i, j] = self.charTrans[labelList[i][j]]


            yield tmpMatrix, resultLabel, seqLen

    def thread_main(self, sess, file):
        for dataX, dataY, seqLen in self.data_iterator(file):
            sess.run(self.enqueue_op, feed_dict={self.dataX: dataX, self.dataY: dataY, self.seqLen: seqLen})


    def start_threads(self, sess, n_threads=1):
        threads = []
        for n in range(n_threads):
            file = open(self.dataFileName, 'rb')
            for i in range(n * 997):
                utility.ReadNextEntry(file)
            t = threading.Thread(target=self.thread_main, args=(sess, file))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads

    def get_max_height(self):
        return self.maxHeight
    def get_max_width(self):
        return self.maxWidth
    def get_num_chars(self):
        return len(self.charDict)
    def get_max_seq_len(self):
        return self.maxSeqLen

    def decode_Num(self, string):
        result = ""
        for i in string:
            if i != -1:
                result += self.charTransInv[i]
        return result
    def get_num_pad(self):
        return -1

#Reads custom data files in parallel.
#All threads start at different offsets in the file.
#uses a fifo queue
class CustomAsynchBatchLoaderFull(object):
    def __init__(self, metaFileName, dataFileName, batch_size, capacity, min_after_dequeue):
        # meta file contains general information about the data in binary.
        # Those are:
        #   all used characters
        #   maximal image height
        #   maximal image width
        #   maximal sequence length

        self.metaFile = open(metaFileName, 'rb')
        self.allChars, self.maxHeight, self.maxWidth, self.maxSeqLen = utility.ReadMetaData(self.metaFile)
        self.allChars += chr(3)
        self.charDict = list(set(self.allChars))
        self.charTrans = dict(zip(self.charDict, np.arange(len(self.charDict))))
        self.charTransInv = dict((v, k) for k, v in self.charTrans.items())
        self.batchSize = batch_size
        self.finished = False
        self.set_reset = False
        self.dataFileName = dataFileName

        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.maxHeight, self.maxWidth])
        self.dataY = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.seqLen = tf.placeholder(dtype=tf.int32, shape=[None])

        self.queue = tf.FIFOQueue(capacity, [tf.float32, tf.int32, tf.int32])

        self.enqueue_op = self.queue.enqueue([self.dataX, self.dataY, self.seqLen])
        self.size_op = self.queue.size()

    def get_inputs(self):
        outputImages, outputLabels, outputSeqLen = list(self.queue.dequeue())
        return outputImages, outputLabels, outputSeqLen

    def get_queue(self):
        return self.queue

    def data_iterator(self, file):
        while True:
            idxs = np.arange(0, self.batchSize)
            #np.random.shuffle(idxs)
            labelList = list()
            dataList = list()
            seqLen = np.zeros(self.batchSize, dtype=np.int32)
            counter = 0
            while counter < self.batchSize or (self.finished is True):
                label, data = utility.ReadNextEntryEnd(file)
                if not (label is None):
                    label += chr(3)
                    seqLen[counter] = len(label)
                    labelList.append(label)
                    dataList.append(data)
                    counter = counter + 1
                else:
                    self.finished = True
                    if self.set_reset:
                        utility.Reset(file)
                        self.finished = False
                        self.set_reset = False


            maxLen = np.amax(seqLen)

            tmpMatrix = np.zeros(shape=(self.batchSize, self.maxHeight, self.maxWidth))

            for i in range(min(self.batchSize, counter)):
                idx = idxs[i]
                shape = dataList[i].shape
                tmpMatrix[idx, :shape[0], :shape[1]] = dataList[i]
            resultLabel = np.full(shape=(self.batchSize, maxLen), fill_value=-1)
            for i in range(min(self.batchSize, counter)):
                idx = idxs[i]
                for j in range(seqLen[i]):
                    resultLabel[idx, j] = self.charTrans[labelList[i][j]]


            yield tmpMatrix, resultLabel, seqLen

    def thread_main(self, sess, file):
        for dataX, dataY, seqLen in self.data_iterator(file):
            sess.run(self.enqueue_op, feed_dict={self.dataX: dataX, self.dataY: dataY, self.seqLen: seqLen})


    def start_threads(self, sess, n_threads=1):
        threads = []
        for n in range(n_threads):
            file = open(self.dataFileName, 'rb')
            #offset for each Thread.
            for _ in range(n * 997):
                utility.ReadNextEntry(file)
            t = threading.Thread(target=self.thread_main, args=(sess, file))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads

    def get_max_height(self):
        return self.maxHeight
    def get_max_width(self):
        return self.maxWidth
    def get_num_chars(self):
        return len(self.charDict)
    def get_max_seq_len(self):
        return self.maxSeqLen

    def decode_Num(self, string, lb):
        result = ""
        for i in range(string.shape[0]):
            if lb[i] != -1:
                result += self.charTransInv[np.argmax(string[i])]
        return result

    def decode_label(self, string):
        result = ""
        for i in range(string.shape[0]):
            if string[i] != -1:
                result += self.charTransInv[string[i]]
        return result

    def decode_Num_Trans(self, string, translation):
        result = ""
        for i in string:
            if i != -1:
                if i in translation:
                    result += translation[i]
                if translation[i] == "":
                    break
        return result

    def set_translation(self, dictionary):
        tmpDictionary = dict()
        maxValue = 0
        minValue = 9999
        for value in self.charTrans.values():
            if value > maxValue:
                maxValue = value
            if minValue > value:
                minValue = value

        for key, value in dictionary.items():
            tmpDictionary[key] = value
        for key, value in self.charTrans.items():
            if key not in tmpDictionary.values():
                for i in range(minValue, maxValue + 1):
                    if i not in tmpDictionary.keys():
                        tmpDictionary[i] = key
                        break

        self.charTransInv = tmpDictionary
        self.charTrans = dict((v, k) for k, v in self.charTransInv.items())

    def get_num_pad(self):
        return -1
    def end(self, sess):
        return self.finished and (sess.run(self.size_op) == 0)
    def reset(self):
        self.set_reset = True
    def get_queue_size(self, sess):
        return sess.run(self.size_op)
    def is_finished(self):
        return self.finished

# class CustomAsynchSparseBatchLoader(object):
#     def __init__(self, metaFileName, dataFileName, batch_size, capacity, min_after_dequeue):
#         self.metaFile = open(metaFileName, 'rb')
#         self.allChars, self.maxHeight, self.maxWidth, self.maxSeqLen = utility.ReadMetaData(self.metaFile)
#         self.allChars += chr(3)
#         self.charDict = list(set(self.allChars))
#         self.charTrans = dict(zip(self.charDict, np.arange(len(self.charDict))))
#         self.charTransInv = dict((v, k) for k, v in self.charTrans.items())
#         self.batchSize = batch_size
#
#         self.dataFileName = dataFileName
#
#         self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.maxHeight, self.maxWidth])
#         self.indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])
#         self.dataY = tf.placeholder(dtype=tf.int32, shape=[None])
#         self.shape = tf.placeholder(dtype=tf.int64, shape=[2])
#         self.seqLen = tf.placeholder(dtype=tf.int32, shape=[None])
#
#         self.queue = tf.RandomShuffleQueue(dtypes=[tf.float32, tf.int64, tf.int32, tf.int64, tf.int32],
#                                            capacity=capacity,
#                                            min_after_dequeue=min_after_dequeue)
#
#         self.enqueue_op = self.queue.enqueue([self.dataX, self.indices, self.dataY, self.shape, self.seqLen])
#
#     def get_inputs(self):
#         outputImages, indices, labels, shape, outputSeqLen = list(self.queue.dequeue())
#         sparse_labels = tf.SparseTensor(indices=indices, values=labels, dense_shape=shape)
#         return outputImages, sparse_labels, outputSeqLen
#
#     def data_iterator(self, file):
#         while True:
#             labelList = list()
#             dataList = list()
#             seqLen = np.zeros(self.batchSize, dtype=np.int32)
#             for i in range(self.batchSize):
#                 label, data = utility.ReadNextEntryEnd(file)
#                 if not (label is None):
#                     label += chr(3)
#                     seqLen[i] = len(label)
#                     labelList.append(label)
#                     dataList.append(data)
#                 else:
#                     self.finished = True
#                     if self.set_reset:
#                         utility.Reset(file)
#                         self.finished = False
#                         self.set_reset = False
#
#             maxLen = np.amax(seqLen)
#
#             tmpMatrix = np.zeros(shape=(self.batchSize, self.maxHeight, self.maxWidth))
#
#             for i in range(self.batchSize):
#                 shape = dataList[i].shape
#                 tmpMatrix[i, :shape[0], :shape[1]] = dataList[i]
#             resultLabel = np.zeros(shape=(np.sum(seqLen)), dtype=np.int32)
#             indices = np.zeros(shape=(np.sum(seqLen), 2), dtype=np.int64)
#             offset = 0
#             for i in range(self.batchSize):
#                 for j in range(seqLen[i]):
#                     resultLabel[offset + j] = self.charTrans[labelList[i][j]]
#                     indices[offset + j] = [i,j]
#                 offset += seqLen[i]
#             yield tmpMatrix, indices, resultLabel, [self.batchSize, maxLen], seqLen
#
#     def thread_main(self, sess, file):
#         for dataX, indices, dataLabel, shape, seqLen in self.data_iterator(file):
#             sess.run(self.enqueue_op, feed_dict={self.dataX: dataX, self.indices: indices ,self.dataY: dataLabel, self.shape: shape, self.seqLen: seqLen})
#
#
#     def start_threads(self, sess, n_threads=1):
#         threads = []
#         for n in range(n_threads):
#             file = open(self.dataFileName, 'rb')
#             for i in range(n * 997):
#                 utility.ReadNextEntry(file)
#             t = threading.Thread(target=self.thread_main, args=(sess, file))
#             t.daemon = True
#             t.start()
#             threads.append(t)
#         return threads
#
#     def get_max_height(self):
#         return self.maxHeight
#     def get_max_width(self):
#         return self.maxWidth
#     def get_num_chars(self):
#         return len(self.charDict)
#     def get_max_seq_len(self):
#         return self.maxSeqLen
#
#     def decode_Num(self, string):
#         result = ""
#         for i in string:
#             if i != -1:
#                 result += self.charTransInv[i]
#         return result
#     def get_num_pad(self):
#         return -1

class CustomAsynchBatchLoaderBucket(object):
    def __init__(self, metaFileName, dataFileName, batch_size, capacity, min_after_dequeue, num_buckets):
        self.metaFile = open(metaFileName, 'rb')
        self.allChars, self.maxHeight, self.maxWidth, self.maxSeqLen = utility.ReadMetaData(self.metaFile)
        self.allChars += chr(3)
        self.charDict = list(set(self.allChars))
        self.charTrans = dict(zip(self.charDict, np.arange(len(self.charDict))))
        self.charTransInv = dict((v, k) for k, v in self.charTrans.items())
        self.batchSize = batch_size

        self.num_buckets = num_buckets

        self.dataFileName = dataFileName

        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.maxHeight, None])
        self.dataY = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.seqLen = tf.placeholder(dtype=tf.int32, shape=[None])

        self.queue = tf.RandomShuffleQueue(dtypes=[tf.float32, tf.int32, tf.int32],
                                           capacity=capacity,
                                           min_after_dequeue=min_after_dequeue)

        self.enqueue_op = self.queue.enqueue([self.dataX, self.dataY, self.seqLen])

    def get_inputs(self):
        outputImages, outputLabels, outputSeqLen = list(self.queue.dequeue())
        return outputImages, outputLabels, outputSeqLen

    def get_queue(self):
        return self.queue

    def data_iterator(self, file):
        labelListList = list()
        dataListList = list()
        seqLenListList = list()
        baseSize = math.ceil(self.maxWidth / self.num_buckets)
        for i in range(self.num_buckets):
            labelListList.append(list())
            dataListList.append(list())
            seqLenListList.append(list())

        while True:

            done = False
            # idxs = np.arange(0, self.batchSize)
            # #np.random.shuffle(idxs)
            # labelList = list()
            # dataList = list()
            # seqLen = np.zeros(self.batchSize, dtype=np.int32)
            bucket = 0
            while not done:
                label, data = utility.ReadNextEntry(file)
                label += chr(3)
                shape = data.shape
                bucket = math.ceil(shape[1]/baseSize)-1

                seqLenListList[bucket].append(len(label))
                labelListList[bucket].append(label)
                dataListList[bucket].append(data)

                numElements = len(dataListList[bucket])
                if numElements == self.batchSize:
                    done = True

            seqLen = np.array(seqLenListList[bucket], dtype=np.int32)
            maxLen = np.amax(seqLen)
            widths = np.zeros(self.batchSize, dtype=np.int32)
            for i in range(self.batchSize):
                widths[i] = dataListList[bucket][i].shape[1]
            maxWidth = np.amax(widths)

            tmpMatrix = np.zeros(shape=(self.batchSize, self.maxHeight, maxWidth+40))

            dataList = dataListList[bucket]
            labelList = labelListList[bucket]

            for i in range(self.batchSize):
                shape = dataList[i].shape
                tmpMatrix[i, :shape[0], :shape[1]] = dataList[i]
            resultLabel = np.full(shape=(self.batchSize, maxLen), fill_value=-1)
            for i in range(self.batchSize):
                for j in range(seqLen[i]):
                    resultLabel[i, j] = self.charTrans[labelList[i][j]]

            seqLenListList[bucket].clear()
            labelListList[bucket].clear()
            dataListList[bucket].clear()

            yield tmpMatrix, resultLabel, seqLen

    def thread_main(self, sess, file):
        for dataX, dataY, seqLen in self.data_iterator(file):
            sess.run(self.enqueue_op, feed_dict={self.dataX: dataX, self.dataY: dataY, self.seqLen: seqLen})


    def start_threads(self, sess, n_threads=1):
        threads = []
        for n in range(n_threads):
            file = open(self.dataFileName, 'rb')
            for i in range(n * 997):
                utility.ReadNextEntry(file)
            t = threading.Thread(target=self.thread_main, args=(sess, file))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads

    def get_max_height(self):
        return self.maxHeight
    def get_max_width(self):
        return self.maxWidth
    def get_num_chars(self):
        return len(self.charDict)
    def get_max_seq_len(self):
        return self.maxSeqLen

    def decode_Num(self, string):
        result = ""
        for i in string:
            if i != -1:
                result += self.charTransInv[i]
        return result

    def decode_Num_Trans(self, string, translation):
        result = ""
        for i in string:
            if i != -1:
                if i in translation:
                    result += translation[i]
        return result

    def get_num_pad(self):
        return -1



class CustomAsynchBatchLoaderBucketFull(object):
    def __init__(self, metaFileName, dataFileName, batch_size, capacity, min_after_dequeue, num_buckets):
        self.metaFile = open(metaFileName, 'rb')
        self.allChars, self.maxHeight, self.maxWidth, self.maxSeqLen = utility.ReadMetaData(self.metaFile)
        self.allChars += chr(3)
        self.charDict = list(set(self.allChars))
        self.charTrans = dict(zip(self.charDict, np.arange(len(self.charDict))))
        self.charTransInv = dict((v, k) for k, v in self.charTrans.items())
        self.batchSize = batch_size
        self.finished = False
        self.num_buckets = num_buckets
        self.set_reset = False
        self.dataFileName = dataFileName

        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, self.maxHeight, None])
        self.dataY = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.seqLen = tf.placeholder(dtype=tf.int32, shape=[None])

        self.queue = tf.FIFOQueue(capacity, dtypes=[tf.float32, tf.int32, tf.int32])

        self.enqueue_op = self.queue.enqueue([self.dataX, self.dataY, self.seqLen])
        self.size_op = self.queue.size()

    def get_inputs(self):
        outputImages, outputLabels, outputSeqLen = list(self.queue.dequeue())
        return outputImages, outputLabels, outputSeqLen

    def get_queue(self):
        return self.queue

    def data_iterator(self, file):
        labelListList = list()
        dataListList = list()
        seqLenListList = list()
        baseSize = math.ceil(self.maxWidth / self.num_buckets)
        for i in range(self.num_buckets):
            labelListList.append(list())
            dataListList.append(list())
            seqLenListList.append(list())

        while True:

            done = False
            # idxs = np.arange(0, self.batchSize)
            # #np.random.shuffle(idxs)
            # labelList = list()
            # dataList = list()
            # seqLen = np.zeros(self.batchSize, dtype=np.int32)
            bucket = 0
            while not done or (self.finished is True):
                label, data = utility.ReadNextEntryEnd(file)
                if not(label is None):
                    label += chr(3)
                    shape = data.shape
                    bucket = math.ceil(shape[1] / baseSize) - 1

                    seqLenListList[bucket].append(len(label))
                    labelListList[bucket].append(label)
                    dataListList[bucket].append(data)

                    numElements = len(dataListList[bucket])
                    if numElements == self.batchSize:
                        done = True
                else:
                    self.finished = True
                    if self.set_reset:
                        utility.Reset(file)
                        self.finished = False
                        self.set_reset = False

            seqLen = np.array(seqLenListList[bucket], dtype=np.int32)
            maxLen = np.amax(seqLen)
            widths = np.zeros(self.batchSize, dtype=np.int32)
            for i in range(self.batchSize):
                widths[i] = dataListList[bucket][i].shape[1]
            maxWidth = np.amax(widths)

            tmpMatrix = np.zeros(shape=(self.batchSize, self.maxHeight, maxWidth+40))

            dataList = dataListList[bucket]
            labelList = labelListList[bucket]

            for i in range(self.batchSize):
                shape = dataList[i].shape
                tmpMatrix[i, :shape[0], :shape[1]] = dataList[i]
            resultLabel = np.full(shape=(self.batchSize, maxLen), fill_value=-1)
            for i in range(self.batchSize):
                for j in range(seqLen[i]):
                    resultLabel[i, j] = self.charTrans[labelList[i][j]]

            seqLenListList[bucket].clear()
            labelListList[bucket].clear()
            dataListList[bucket].clear()

            yield tmpMatrix, resultLabel, seqLen

    def thread_main(self, sess, file):
        for dataX, dataY, seqLen in self.data_iterator(file):
            sess.run(self.enqueue_op, feed_dict={self.dataX: dataX, self.dataY: dataY, self.seqLen: seqLen})

    def start_threads(self, sess, n_threads=1):
        threads = []
        for n in range(n_threads):
            file = open(self.dataFileName, 'rb')
            for i in range(n * 997):
                utility.ReadNextEntry(file)
            t = threading.Thread(target=self.thread_main, args=(sess, file))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads

    def get_max_height(self):
        return self.maxHeight

    def get_max_width(self):
        return self.maxWidth

    def get_num_chars(self):
        return len(self.charDict)

    def get_max_seq_len(self):
        return self.maxSeqLen

    def decode_Num(self, string):
        result = ""
        for i in string:
            if i != -1:
                result += self.charTransInv[i]
        return result

    def decode_Num_Trans(self, string, translation):
        result = ""
        for i in string:
            if i != -1:
                if i in translation:
                    result += translation[i]
        return result

    def get_num_pad(self):
        return -1
    def end(self, sess):
        return self.finished and (sess.run(self.size_op) == 0)
    def reset(self):
        self.set_reset = True
    def get_queue_size(self, sess):
        return sess.run(self.size_op)
    def is_finished(self):
        return self.finished
        # loader = CustomAsynchBatchLoaderBucket('metaFile.dat', 'D:/MLProjects/OCR2/trainData.dat', 10, 200, 20, 3)
# file = open('D:/MLProjects/OCR2/trainData.dat', 'rb')
#
# for data, label, seqLen in loader.data_iterator(file):
#     print(data.shape)