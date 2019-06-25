# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:28:04 2018

@author: srikar
"""

import pandas as pd
import numpy as np

#Computes dot product between two lists
def dot(K, L):
    if len(K) != len(L):
      return 0
    return sum(i[0] * i[1] for i in zip(K, L))

def subtract(l1, l2):
    return [a - b for a, b in zip(l1, l2)]
    
def add(l1, l2):
    return [a + b for a, b in zip(l1, l2)]

#parses text file to splut it into list of tuples
def create_input(file):
    data = open(file, 'r')
    res = []
    for line in data.readlines():
        c = line.replace('im', '').split('\t')[1:3]
        x = (c[0], c[1])
        res.append(x)
    res = [x for x in res if x != ('', '')]
    return res

#converts the xi from the tuple pair input (xi, yi) into a list of integers
#consisting of 0's and 1's, this is done to make doing the dot product easier
def convert_to_int(l):
    res = []
    for i in range(len(l)):
        x = list(l[i][0])
        x = [float(i) for i in x]
        res.append((x,l[i][1]))
    return res


#classification algorithm
#has 26 classes for each of the letter classes
#Does T iterations and goes through the input set D
#Predicts class and then compares it to actual class
#if incorrect weight is updated
def multi_class_classifier(D, k, T):
    classification = {1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6: 'f', 7:'g',
             8:'h', 9:'i', 10:'j', 11:'k', 12:'l', 13:'m', 14:'n',
             15:'o', 16:'p', 17:'q', 18:'r', 19:'s', 20: 't', 21:'u',
             22:'v', 23:'w', 24:'x', 25:'y', 26:'z'}
    w = [[0]*128 for _ in range(k)] #initialize weights
    n = 1 #learning rate
    mistakes = []
    accuracy = [] 
    m = 0
    #T iterations
    for i in range(T):
        count = 1
        tot = 1
        #Through input D
        for j in range(len(D)):
            x_t = D[i][0] #input value
            y_t = D[i][1] #correct output
            possibilities = [[0] for _ in range(k)] #the 26 possibile classifications
            for z in range(len(w)-1):
                possibilities[z] = dot(w[z], x_t) #get dot product between each of the 26 weights and input
            possibilities = possibilities[:len(possibilities)-1] #remove the last value of possibilities (extra)
            y_hat_t = classification[possibilities.index(max(possibilities))+1] #get index of max dot product value, the corresponding letter is the predicted classfication y_hat_t
            if y_t != y_hat_t: #if not equal update weights

                w[possibilities.index(max(possibilities))] = add(w[possibilities.index(max(possibilities))], n*x_t)
                w_index = list(classification.keys())[list(classification.values()).index(y_t)]
                w[w_index-1] = subtract(w[w_index-1],n*x_t)
                m+=1 #mistakes
            else:
                count+=1
        tot+=1
        accuracy.append(count/tot)
        mistakes.append(m) #return weight, accuracy and mistakes
    return w, accuracy, mistakes

def avg_multi_class_classifier(D, k, T):
    classification = {1:'a', 2:'b', 3:'c', 4:'d', 5:'e', 6: 'f', 7:'g',
             8:'h', 9:'i', 10:'j', 11:'k', 12:'l', 13:'m', 14:'n',
             15:'o', 16:'p', 17:'q', 18:'r', 19:'s', 20: 't', 21:'u',
             22:'v', 23:'w', 24:'x', 25:'y', 26:'z'}
    w = [[0]*128 for _ in range(k)]
    n = 1
    mistakes = []
    accuracy = []
    m = 0
    for i in range(T):
        count = 1
        tot = 1
        cm = 1
        for j in range(len(D)):
            x_t = D[i][0]
            y_t = D[i][1]
            possibilities = [[0] for _ in range(k)]
            for z in range(len(w)-1):
                possibilities[z] = dot((cm*w[z]), x_t) #multiply by lifespan of weight
            possibilities = possibilities[:len(possibilities)-1]
            y_hat_t = classification[possibilities.index(max(possibilities))+1]
            if y_t != y_hat_t:
                w[possibilities.index(max(possibilities))] = add(w[possibilities.index(max(possibilities))],n*x_t)
                w_index = list(classification.keys())[list(classification.values()).index(y_t)]
                w[w_index-1] = subtract(w[w_index-1],n*x_t)
                m+=1
            else:
                count+=1
                cm+=1
        tot+=1
        accuracy.append(count/tot)
        mistakes.append(m)
    return w, accuracy, mistakes

def main():

    train = create_input('ocr_train.txt')
    train = convert_to_int(train)

    train_weight, train_accuracy, train_mistakes = multi_class_classifier(train, 26, 20)
    tot_train_accuracy = np.mean(train_accuracy)
    
    avg_train_weight, avg_train_accuracy, avg_train_mistakes = avg_multi_class_classifier(train, 26, 20)
    avg_tot_train_accuracy = np.mean(train_accuracy)
    
    test = create_input('ocr_test.txt')
    test = convert_to_int(test)

    test_weight, test_accuracy, test_mistakes = multi_class_classifier(test, 26, 20)
    tot_test_accuracy = np.mean(test_accuracy)
    
    avg_test_weight, avg_test_accuracy, avg_test_mistakes = avg_multi_class_classifier(test, 26, 20)
    avg_tot_test_accuracy = np.mean(test_accuracy)
    
    text_file = open("output.txt", "w")
    
    text_file.write('OCR Multi-class Classifier' + '\n')

    
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




