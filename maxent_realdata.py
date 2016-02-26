#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
 Logistic Regression
 
 References :
    - Jason Rennie: Logistic Regression,
   http://qwone.com/~jason/writing/lr.pdf
 
   - DeepLearningTutorials
   https://github.com/lisa-lab/DeepLearningTutorials


'''

import sys
import numpy


numpy.seterr(all='ignore')
 
def sigmoid(x):
    return 1. / (1 + numpy.exp(-x))

def softmax(x):
    e = numpy.exp(x - numpy.max(x))  # prevent overflow
	#x=x.W +b
    ep =0
    ep_total =0
    vec1=numpy.exp(x)
    #res=vec1.T/numpy.sum(vec1,axis=1)
	#return res.T
    if e.ndim == 1:
        res=vec1.T/numpy.sum(vec1,axis=0)
        #print res
        #print e / numpy.sum(e, axis=0)
        return e / numpy.sum(e, axis=0)
    else:  
        res=vec1.T/numpy.sum(vec1,axis=1)
        #print res
        #print e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # ndim = 2

def maxindex(x):
    print numpy.where(x==x.max())[0]
    return numpy.where(x==x.max())[0]

class LogisticRegression(object):
    def __init__(self, input, label, n_in, n_out):
        self.x = input
        self.y = label
        self.W = numpy.zeros((n_in, n_out))  # initialize W 0
        self.b = numpy.zeros(n_out)          # initialize bias 0

        # self.params = [self.W, self.b]

    def train(self, lr=0.1, input=None, L2_reg=0.00):
        if input is not None:
            self.x = input

        # p_y_given_x = sigmoid(numpy.dot(self.x, self.W) + self.b)
        p_y_given_x = softmax(numpy.dot(self.x, self.W) + self.b)
        d_y = self.y - p_y_given_x
        
        self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
        self.b += lr * numpy.mean(d_y, axis=0)
        
        # cost = self.negative_log_likelihood()
        # return cost

    def negative_log_likelihood(self):
        # sigmoid_activation = sigmoid(numpy.dot(self.x, self.W) + self.b)
        sigmoid_activation = softmax(numpy.dot(self.x, self.W) + self.b)

        cross_entropy = - numpy.mean(
            numpy.sum(self.y * numpy.log(sigmoid_activation) +
            (1 - self.y) * numpy.log(1 - sigmoid_activation),
                      axis=1))

        return cross_entropy


    def predict(self, x):
        # return sigmoid(numpy.dot(x, self.W) + self.b)
        return softmax(numpy.dot(x, self.W) + self.b)


def test_lr(learning_rate=0.01, n_epochs=200):
    # training data
    x = numpy.array([[1,1,1,0,0,0],
                     [1,0,1,0,0,0],
                     [0,1,1,0,0,0],
                     [1,1,0,0,0,0],
                     [0,0,1,1,1,0],
                     [0,0,1,1,0,0],
                     [0,0,0,1,1,0],
                     [0,0,0,1,1,1],
                     [0,0,0,0,1,1]])
    #-------------------------------------------reading data
    print "Reading Data..."

    dataset_name = "books"


    trainData = open(dataset_name + '_min_train.features').readlines()
    for i in range(len(trainData)):
        trainData[i] = trainData[i].split()
    trainData = numpy.array(trainData,dtype=int)            #trainfeatures
    trainData.astype(int)
    print trainData

    testlenght = len(trainData[0])
    print testlenght


    validationData = open(dataset_name + '_min_test.features').readlines()
    for i in range(len(validationData)):
        validationData[i] = validationData[i].split()
    validationData = numpy.array(validationData,dtype=int)
    validationData.astype(int)                           #testfeatures
    print validationData


    trainLabels = open(dataset_name + '_min_train.labels').readlines()
    trainLabels = [i.strip() for i in trainLabels]
    tl = numpy.array([0,0,0,0,0],dtype=int)
    for i in trainLabels:
        temp = numpy.zeros(5,dtype=int)
        temp[int(float(i))-1] = 1
        tl = numpy.vstack((tl,temp))
    trainLabels = numpy.array(trainLabels)  
    # trainLabels.astype(int)       
    # print trainLabels
    tl = tl[1:]
    print ">>>>",tl, len(tl)
    trainLabels = tl                                #trainlabels


    # validLabels = open(dataset_name + '_test.labels').readlines()
    # validLabels = [i.strip() for i in validLabels]
    # validLabels = numpy.array(validLabels)         #testlabels

    validLabels = open(dataset_name + '_min_test.labels').readlines()
    validLabels = [i.strip() for i in validLabels]
    vl = numpy.array([0,0,0,0,0],dtype=int)
    vl.astype(int)
    for i in validLabels:
        temp = numpy.zeros(5,dtype=int)
        temp[int(float(i))-1] = 1
        temp.astype(int)
        vl = numpy.vstack((vl,temp)) 
    validLabels = numpy.array(validLabels)         #testlabels
    # print validLabels
    vl = vl[1:]
    vl.astype(int)
    print ">>>>",vl, len(vl)
    validLabels = vl


    print "Read Data..."
    #/-------------------------------------------reading data

    y = numpy.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 0, 1]])


    # construct LogisticRegression
    #feature_len = len(validationData[0])

    classifier = LogisticRegression(input=trainData, label=tl, n_in=testlenght, n_out=5)
    # classifier = LogisticRegression(input=x, label=y, n_in=6, n_out=4)

    # train
    for epoch in xrange(n_epochs):
        classifier.train(lr=learning_rate)
        cost = classifier.negative_log_likelihood()
        # print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost
        learning_rate *= 0.95


    # ------------------------------------------------------------------------------test
    #x = numpy.array([1, 1, 0, 0, 0, 0])
    x = numpy.array([0,0,3,4,0,0])
    x = validationData[2]
    logit = classifier.predict(x)
    print >> sys.stderr, logit      #classifier.predict(x)
    pred_class = maxindex(logit)     #indexign strts from 1
    print "predicted x= ", pred_class
    print "actual class= ", numpy.where(vl[2]==1)[0]
    print "len of logit", logit

    #-----------------------------------------------------------------init accuracy vars
    total = 0.0
    correct = 0.0

    #------------------------------------------------------------------testing all cases
    numberoftestcases = len(validationData)
    for i in xrange(numberoftestcases):
        print "testing for case:", i, "------------------------"
        x = validationData[i]
        logit = classifier.predict(x)
        print >> sys.stderr, logit      #classifier.predict(x)
        pred_class = maxindex(logit)     #indexign strts from 1
        print "predicted x[", i, "]= ", pred_class
        actual_class = numpy.where(vl[i]==1)[0]
        print "actual class[", i, "]= ", actual_class
        if (pred_class == actual_class):
            correct = correct+1
        print "len of logit", logit
        total = total+1

    accuracy = correct/total
    print accuracy

if __name__ == "__main__":
    test_lr()

