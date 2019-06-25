# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 17:41:47 2018

@author: srikar
"""

import pandas as pd
import numpy as np
from string import punctuation
from random import random

def sign(a):
    return bool(a > 0) - bool(a < 0)

def dot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))

def remove_stopwords(file, stopwords):
    with open(file) as text,  open(stopwords) as filter_words:
        st = set(word.lower().rstrip(punctuation+"\n") for word in  filter_words)
        txt = next(text).lower().split()
        out = [word  for word in txt if word not in st]
    return out

def convert_to_feature_vector(vocabulary, data):
    res = []
    f = open(data)
    for line in f.readlines():
        m = [0]*len(vocabulary)
        c = list(line.rstrip('\n').split(' '))
        for i in range(len(c)):
            if c[i] in vocabulary:
                m[vocabulary.index(c[i])] = 1
        
        res.append(m)
    return res


def ready_labels(data):
    labels = list(open(data, 'r'))
    labels = [s.rstrip() for s in labels]
    labels = [int(x) for x in labels]
    return labels




def binary_classifier(D, T, vocabulary, labels):
    w = [0]*len(vocabulary)
    n = 1
    #wise = 0
    #future = 1
    mistakes = []
    m = 0
    accuracy = []
    for i in range(T):
        count = 0
        tot = 1
        b = 0
        for j in range(len(D)):
            if len(w) != 0:
                y_hat_t = sign(dot(w, D[j]) + b)
                if y_hat_t <= 0:
                    w_prev = w
                    change = n*labels[j]*D[j]
                    w[:] = [(x+y) for x,y in zip(w, change)]
                    b+=labels[j]
                    m+=1
                else:
                    count+=1
            tot+=1
        accuracy.append(count/tot)
        mistakes.append(m)
    return w_prev, mistakes, accuracy

def avg_binary_classifier(D, T, vocabulary, labels):
    w = [0]*len(vocabulary)
    n = 1
    #wise = 0
    #future = 1
    mistakes = []
    accuracy = []
    m = 0
    for i in range(T):
        count = 0
        tot = 1
        k = 1
        cm = 0
        b = 0
        beta = 0
        u = 0

        for j in range(len(D)):
            if len(w) != 0:
                w[:] = [float(x)*cm/k for x in w]
                y_hat_t = sign(dot(w, D[j]) + b)
                if y_hat_t <= 0:
                    w_prev = w
                    change = n*labels[j]*D[j]
                    w[:] = [float(x+y) for x,y in zip(w, change)]
                    u = [labels[j]*count*x for x in D[j]]
                    k+=1
                    cm = 1
                    m+=1
                    beta+=labels[j]*count
                    b+=labels[j]
                else:
                    cm+=1
                    count+=1
            tot+=1
        accuracy.append(count/tot)
        mistakes.append(m)
        w_prev = [x-1/count*sum(u) for x in w_prev]
    return w_prev, mistakes, accuracy




def main():
    vocabulary = sorted(remove_stopwords('traindata.txt', 'stoplist.txt'))
    
    feature_vectors_train = convert_to_feature_vector(vocabulary, 'traindata.txt')
    feature_vectors_test = convert_to_feature_vector(vocabulary, 'testdata.txt')
    
    trainlabels = ready_labels('trainlabels.txt')
    train_weight, train_mistakes, train_accuracy = binary_classifier(feature_vectors_train, 20, vocabulary, trainlabels)
    tot_train_accuracy = np.mean(train_accuracy)
    
    avg_train_weight, avg_train_mistakes, avg_train_accuracy = avg_binary_classifier(feature_vectors_train, 20, vocabulary, trainlabels) 
    avg_tot_train_accuracy = np.mean(avg_train_accuracy)
    
    
    
    testlabels = ready_labels('testlabels.txt')     
    test_weight, test_mistakes, test_accuracy = binary_classifier(feature_vectors_test, 20, vocabulary, testlabels)
    tot_test_accuracy = np.mean(test_accuracy)
    
    avg_test_weight, avg_test_mistakes, avg_test_accuracy = avg_binary_classifier(feature_vectors_test, 20, vocabulary, testlabels)            
    avg_tot_test_accuracy = np.mean(avg_test_accuracy)
    
    text_file = open("output.txt", "w")
    
    text_file.write('Fortune Cookie Binary Classifier' + '\n')

    
    text_file.write('Train Mistakes' + '\n')
    for item in train_mistakes:
      text_file.write("%s\n" % item)
      
    text_file.write('Test Mistakes' + '\n')
    for item in test_mistakes:
      text_file.write("%s\n" % item)
    
    text_file.write('Train Accuracy' + '\n')
    for item in train_accuracy:
      text_file.write("%s\n" % item)
    
    text_file.write('Test Accuracy' + '\n')
    for item in test_accuracy:
      text_file.write("%s\n" % item)
    
    text_file.write('Standard Perceptron Total Train Accuracy' + '\n')
    text_file.write(str(tot_train_accuracy) + '\n')
    
    text_file.write('Standard Perceptron Total Test Accuracy' + '\n')
    text_file.write(str(tot_test_accuracy) + '\n')
    
    
    text_file.write('Average Perceptron Total Train Accuracy' + '\n')
    text_file.write(str(avg_tot_train_accuracy) + '\n')
    
    text_file.write('Average Perceptron Total Test Accuracy' + '\n')
    text_file.write(str(avg_tot_test_accuracy) + '\n')
    
    text_file.close()
    
if __name__ == '__main__':
    main()
