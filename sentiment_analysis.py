import tensorflow as tf
import numpy as np
from random import randint
import re
import pandas as pd
wordsList = np.load('wordsList.npy')

"""
0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive
"""

tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

print(len(wordsList))
print(wordVectors.shape)
numWords = []


maxSeqLength = 20
numDimensions = 300 #Dimensions for each word vector
batchSize = 24
lstmUnits = 64
numClasses = 5
iterations = 150000

num_negatives = 0
num_positives = 0
num_neutral = 0
num_somewhatpositive = 0
num_somewhatnegative = 0

numReviews = 0

positiveFiles_labels = []
negativeFiles_labels = []
neutral_labels = []
somewhat_positiveFiles_labels = []
somewhat_negativeFiles_labels= []

labels_reviews = []

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


def getTrainBatch(ids):
    """
    :return:numpy array arr, numpy array of labels
    """

    labels = []
    arr = np.zeros([batchSize, maxSeqLength])

    for i in range(batchSize):
        num = randint(1,numReviews)
        arr[i] = ids[num-1:num]
        labels.append(labels_reviews[num-1:num][0])
    return arr,np.asarray(labels)



def getData():
    testData_path = r'DATA\test\test.tsv'
    trainData_path = r'DATA\train\train.tsv'

    def getData(path):
        pandaDataframe = pd.DataFrame.from_csv(path, sep='\t')

        return pandaDataframe
    return getData(trainData_path),getData(testData_path)

def prepareData(training_data):

    positiveFiles = training_data[training_data['Sentiment'] == 4]
    num_positives = len(positiveFiles)

    negativeFiles = training_data[training_data['Sentiment'] == 0]
    num_negatives = len(negativeFiles)

    neutral = training_data[training_data['Sentiment'] == 2]
    num_neutral = len(neutral)

    somewhat_positiveFiles = training_data[training_data['Sentiment'] == 3]
    num_somewhatpositive = len(somewhat_positiveFiles)

    somewhat_negativeFiles = training_data[training_data['Sentiment'] == 1]
    num_somewhatnegative = len(somewhat_negativeFiles)

    return positiveFiles,negativeFiles,neutral,somewhat_positiveFiles,somewhat_negativeFiles


def countWords(dataFrame):
    global numWords
    for pf in dataFrame['Phrase']:
        line = pf
        counter = len(line.split())
        numWords.append(counter)

def makeIDSforEachType(dataFrame,fileCounter,ids,type):

    global positiveFiles_labels
    global somewhat_positiveFiles_labels
    global somewhat_negativeFiles_labels
    global negativeFiles_labels
    global neutral_labels
    global labels_reviews

    for pf in dataFrame['Phrase']:

        if type == 4:
            positiveFiles_labels .append([0,0,0,0,1])
            labels_reviews.append([0,0,0,0,1])
        elif type ==3:
            somewhat_positiveFiles_labels.append([0,0,0,1,0])
            labels_reviews.append([0,0,0,1,0])
        elif type == 2:
            neutral_labels.append([0, 0, 1, 0, 0])
            labels_reviews.append([0, 0, 1, 0, 0])
        elif type == 1:
            somewhat_negativeFiles_labels.append([0, 1, 0, 0, 0])
            labels_reviews.append([0, 1, 0, 0, 0])
        else:
            negativeFiles_labels.append([1, 0, 0, 0, 0])
            labels_reviews.append([1, 0, 0, 0, 0])

        indexCounter = 0
        line = pf
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = 399999  # Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter >= maxSeqLength:
                break
        fileCounter = fileCounter + 1

    return fileCounter,ids


def makeIDSMatrix(positiveFiles,negativeFiles,somewhat_positiveFiles,somewhat_negativeFiles,neutral):

    numFiles  = len (numWords)
    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    fileCounter,ids = makeIDSforEachType(positiveFiles,fileCounter,ids,4)

    fileCounter,ids = makeIDSforEachType(somewhat_positiveFiles,fileCounter,ids,3)

    fileCounter,ids= makeIDSforEachType(somewhat_negativeFiles,fileCounter,ids,1)

    fileCounter,ids = makeIDSforEachType(negativeFiles,fileCounter,ids,0)

    fileCounter,ids = makeIDSforEachType(neutral,fileCounter,ids,2)


    np.save('idsMatrix', ids)


class SentimentAnalysisData(object):

    def __init__(self):

        self.labels = tf.placeholder(tf.float32, [batchSize, numClasses])
        self.input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

        self.data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]), dtype=tf.float32)
        self.data = tf.nn.embedding_lookup(wordVectors, self.input_data)

        self.lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        self.lstmCell = tf.contrib.rnn.DropoutWrapper(cell=self.lstmCell, output_keep_prob=0.75)
        self.value, _ = tf.nn.dynamic_rnn(self.lstmCell, self.data, dtype=tf.float32)

        self.weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
        self.bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
        self.value = tf.transpose(self.value, [1, 0, 2])
        self.last = tf.gather(self.value, int(self.value.get_shape()[0]) - 1)
        self.prediction = (tf.matmul(self.last, self.weight) + self.bias)

        self.correctPred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

import datetime


def countWords_in_trainingData():
    global numWords
    global numReviews
    training_data,test_data = getData()
    numReviews = len(training_data)
    result = training_data.sort_values(by=['Sentiment'])
    training_data = result

    positiveFiles, negativeFiles, neutral, somewhat_positiveFiles, somewhat_negativeFiles = prepareData(training_data)
    countWords(positiveFiles)
    countWords(somewhat_positiveFiles)
    countWords(somewhat_negativeFiles)
    countWords(negativeFiles)
    countWords(neutral)
    numFiles = len(numWords)
    print('The total number of files is', numFiles)
    print('The total number of words in the files is', sum(numWords))
    print('The average number of words in the files is', sum(numWords) / len(numWords))

    return positiveFiles, negativeFiles, neutral, somewhat_positiveFiles, somewhat_negativeFiles






import os

def batchTraining(sentimentAnalysisData,ids,sess,logdir,saver):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir, sess.graph)


    for i in range(iterations):
        # Next Batch of reviews

        nextBatch, nextBatchLabels = getTrainBatch(ids)
        feed_dict = {sentimentAnalysisData.input_data: nextBatch,
                     sentimentAnalysisData.labels: nextBatchLabels}

        acc, l, _ = sess.run(
            [sentimentAnalysisData.accuracy, sentimentAnalysisData.loss, sentimentAnalysisData.optimizer],
            feed_dict=feed_dict)
        if (i % 100 == 0):
            print("Minibatch loss at step %d: %3ff" % (i, l))
            print("Minibatch accuracy: %.3f%%" % acc)

        # Write summary to Tensorboard
        if (i % 50 == 0):
            feed_dict = {sentimentAnalysisData.input_data: nextBatch,
                         sentimentAnalysisData.labels: nextBatchLabels}

            summary = sess.run(merged, feed_dict)
            writer.add_summary(summary, i)

        # Save the network every 10,000 training iterations
        if (i % 10000 == 0 and i != 0):
            save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
            print("saved to %s" % save_path)
    writer.close()

def main():
    positiveFiles, negativeFiles, neutral, somewhat_positiveFiles, somewhat_negativeFiles = countWords_in_trainingData()

    makeIDSMatrix(positiveFiles, negativeFiles, somewhat_positiveFiles, somewhat_negativeFiles, neutral)

    ids = np.load('idsMatrix.npy')
    graph = tf.get_default_graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement
        )

        sess = tf.Session(config=session_conf)

        with sess.as_default():

            sentimentAnalysisData = SentimentAnalysisData()
            tf.summary.scalar('Loss', sentimentAnalysisData.loss)
            tf.summary.scalar('Accuracy', sentimentAnalysisData.accuracy)


            logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

            checkpoint_dir = "models"

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

            print("Training begins...")

            if checkpoint_file is None:
                # Initialize all variables
                print("Starting new session")
                sess.run(tf.initialize_all_variables())

                batchTraining(sentimentAnalysisData, ids, sess,logdir,saver)


            else:
                print("Loading checkpoint ......")
                saver.restore(sess, checkpoint_file)
                print("Model restored.")
                batchTraining(sentimentAnalysisData, ids, sess, logdir,saver)


if __name__=="__main__":
    main()