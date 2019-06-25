# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:47:34 2018

@author: srikar
"""
from __future__ import with_statement
from numpy import *
import pandas as pd
from operator import itemgetter

def loadDataSet(data=None):
    return pd.read_csv(data, sep = ' ', error_bad_lines=False)

def createCandidateSet(data):
    C1 = []
    for transaction in data:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scanData(dataset, Ck, support):
    ssCount = {}
    
    for tID in dataset:
        for candidate in Ck:
            if candidate.issubset(tID):
                if not candidate in ssCount:
                    ssCount[candidate] = 1
                else:
                    ssCount[candidate]+=1
#    numItems = float(len(dataset))
    res = []
    supportData ={}
    for key in ssCount:
        #Support is a proportion; the occurence of the item in relation to the data set
#        currSupport = ssCount[key]/numItems
        currSupport = ssCount[key]
        if currSupport >= support:
            res.insert(0, key)
        supportData[key] = currSupport
    return res, supportData

def aprioriHelper(Lk, k): #creates candidate itemsets
    res = []

    freqItemLen = len(Lk)
    for i in range(freqItemLen):
        for j in range(i+1, freqItemLen):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()

            if L1 == L2:
                res.append(Lk[i] | Lk[j])
    return res

def apriori(dataset, minSupport=100):
    C1 = createCandidateSet(dataset)
    D = list(map(set, dataset))
    L1, supportData = scanData(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriHelper(L[k-2], k)
        Lk, supportK = scanData(D, Ck, minSupport) #scan dataset to get frequent items sets, now the itemsets are bigger
        supportData.update(supportK)
        L.append(Lk)
        k+=1
        
    return L, supportData

def generateRules(L, supportData, conf = 0.7): #support data is data on each item sets support, comes from scanData
    rules = [] #takes tuples of associations, consequences, and confidence
    for i in range(1, len(L)): #get itemsets with number of items >=2
        for freq in L[i]:
            association = [frozenset([item]) for item in freq]
            if i > 1:
                rulesFromConsequences(freq, association, supportData, rules, conf)
            else:
                calculateConfidence(freq, association, supportData, rules, conf)
    return rules

def calculateConfidence(freq, association, supportData, rules, conf=0.7):
    filteredAssociations = []
    for consequence in association:
        #confidence(I -> J) = Support(I U J)/Support(I)
        confidence = supportData[freq]/supportData[freq - consequence] #calculate confidence
        if confidence >= conf:
            rules.append((freq-consequence, consequence, confidence))
            filteredAssociations.append(consequence)
    return filteredAssociations

def rulesFromConsequences(freq, association, supportData, rules, conf=0.7):
    #generate more rules when frequent itemsets become larger
    a_len = len(association[0])
    if (len(freq) > (a_len+1)): #try to merge into a bigger itemset that is frequent
        association_p1 = aprioriHelper(association, a_len+1) #create association+1 new candidates- create bigger itemset and get more candidates for association rules
        association_p1 = calculateConfidence(freq, association_p1, supportData, rules, conf)
        if len(association_p1) > 1: #need to have at least two sets in order to merge
            rulesFromConsequences(freq, association_p1, supportData, rules, conf) #recursively call to build bigger itemset and get more rules
            
 
def main():
    
    

    dataset = [line.split() for line in open('browsingdata.txt')]
    L, supportData = apriori(dataset, minSupport=100)

    
    
    rules = generateRules(L, supportData, conf=0)
    rules = sorted(rules, key = itemgetter(2), reverse=True)
    triples = []
    doubles = []
    i = 0
    while len(triples) < 5:
        if i == len(rules):
            break
        if len(rules[i][1]) == 2:
            triples.append(rules[i])
        i+=1
    
    j = 0
    while len(doubles) < 5:
        if len(rules[j][1]) == 1:
            doubles.append(rules[j])
        j+=1

        
    output_A = [['OUTPUT A']]
    output_B = [['OUTPUT B']]
    
    double_format = '{} {} {:0.4f}'
    for double in doubles:
        (first,), (second,), third = double
        output_A.append([double_format.format(first, second, third)])
    
    triple_format = '{} {} {} {:0.4f}'
    for triple in triples:
        (first,), (second1,second2), third = triple
        output_B.append([triple_format.format(first, second1, second2, third)])

    with open('output.txt', 'w') as f:
        for _list in output_A:
            for _string in _list:
                f.write(str(_string) + '\n')
    with open('output.txt', 'a') as f:
        for _list in output_B:
            for _string in _list:
                f.write(str(_string) + '\n')

    
if __name__ == '__main__':
    main()