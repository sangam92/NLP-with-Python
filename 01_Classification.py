#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#import packages
import os
import unicodedata
import pandas as pd
import numpy as np
import string
import nltk
import itertools
from itertools import chain
from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#change working directory to the working directory
print(os.getcwd())
os.chdir("/Users/eotoke/Documents/08.Text Mining/CA")
print(os.getcwd())

#read in the dataset into dataframe data
train_data=pd.read_excel("MsiaAccidentCases.xlsx")

#train_data['Cause']

colnames=['id','Title','Case Details','Keywords','Remarks']
test_data=pd.read_excel("osha.xlsx",header=None,names=colnames)
test_data.columns.values

stemmer = SnowballStemmer("english")

#data.iloc[:,2]
#getting only the causes
train_dataStr=train_data['Title Case']+'. '+train_data['Summary Case']
test_dataStr=test_data['Title'].astype(str)+'. '+test_data['Case Details']
text=pd.concat([train_dataStr,test_dataStr],axis=0)

def clean(vtext):
    #remove unicode encodings
    rtext=unicodedata.normalize('NFKD', vtext).encode('ascii','ignore')
    #remove punctuations. potential change if we were to break by sentence first to do mwe
    rtext=rtext.translate(string.maketrans("",""), string.punctuation)    
    #strip empty space around the sides and set to lowercase
    rtext=rtext.strip().lower()
    return rtext
    
def clean_number(vtext):
    #remove number
    rtext=re.sub(r'[0-9]','',vtext)
    rtext=re.sub(r'[Yy]ear','',rtext)
    return rtext
    
#cleaning the text for each case
cleaned_text=[clean(each_case) for each_case in text]
cleaned_text=[clean_number(each_case) for each_case in cleaned_text]

#creating a simple tfidf with just unigram
#remove stopwords
stopwords = nltk.corpus.stopwords.words('english')

#tokenize each word and stem 
tokenized_text=[nltk.word_tokenize(each_case) for each_case in cleaned_text]
tokenized_text=[[stemmer.stem(word) for word in each_case if word not in stopwords] for each_case in tokenized_text]

tot_text=list(chain.from_iterable(tokenized_text))
fdist=FreqDist(tot_text)
wordList=fdist.values()
wordArray=np.array(wordList)
print '50% quantile word count of',np.percentile(wordArray,50)
print fdist.most_common(30)
#plotting fdist on a cumulative chart
fdist.plot(30,cumulative=True)
#plotting fdist on a non cumulative chart
fdist.plot(30)
print 'seldom appearing words:',fdist.hapaxes()

tfidf_text=[]
for each_case in tokenized_text:
    tfidf_text.append(' '.join(word for word in each_case))
#tfidf_text 

#create a tfidf vectorizer to convert the text into tfidf
tfidf_vectorizer=TfidfVectorizer(min_df=10,max_df=1.0)
tfidf=tfidf_vectorizer.fit_transform(tfidf_text)

feature_names=tfidf_vectorizer.get_feature_names()

#examining each feature in the document and also their corresponding tfidf
for col in tfidf.nonzero()[1]:
    print feature_names[col], ' - ', tfidf[0, col], ' - ', tfidf.indices[col]

#exporting the tfidf
import scipy
scipy.io.mmwrite("tf_idf.mtx.txt", tfidf, comment='', field=None, precision=None)

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    svm.fit(X, y)
    return svm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

#getting training data and labels from the tfidf and train_data
X_all=tfidf[:235]
y_all=train_data.iloc[:,0]

#doing a 10 fold cross validation with rep=10
import random
#import math
#partition of each fold
cvPartition=[24,48,72,96,120,143,166,189,212,235]

#creating index of records to shuffle in the reps later
fullIndex=range(0,235)
random.seed(42)
inSampleAcc=[]
predAcc=[]
for n in xrange(0,10):
    random.shuffle(fullIndex)
    print 'Cross Validation run: ',n+1
    for i in xrange(0,10):
        print 'k-fold: ',i+1
        #trainIndex=random.sample(xrange(235),int(math.floor(0.9*235)))
        #testIndex=list(set(xrange(235))-set(trainIndex))
        if (i==0):
            testIndex=fullIndex[:cvPartition[0]]
        else:
            testIndex=fullIndex[cvPartition[i-1]:cvPartition[i]]
        trainIndex=list(set(fullIndex)-set(testIndex))
        X_train=X_all[trainIndex]
        y_train=y_all[trainIndex]
        svm = train_svm(X_train, y_train)
        X_test=X_all[testIndex]
        y_test=y_all[testIndex]
        inSampleAcc.append(svm.score(X_train,y_train))
        predAcc.append(svm.score(X_test,y_test))

print 'In Sample Classification Accuracy: ',sum(inSampleAcc)/len(inSampleAcc)
print 'Out of Sample Classification Accuracy: ',sum(predAcc)/len(predAcc)

#In Sample Classification Accuracy:  0.996169856031
#Out of Sample Classification Accuracy:  0.706648550725


#out of sample classification accuracy is not very high due to 
#random sampling not taking into consideration unbalanced classes

#ensemble of classifiers
# Decision Tree Classifier
def train_dtc(X, y):
    """
    Create and train the Decision Tree Classifier.
    """
    dtc = DecisionTreeClassifier()
    dtc.fit(X, y)
    return dtc

# K-Nearest Neighbour Classifier
def train_knn(X, y, n, weight):
    """
    Create and train the k-nearest neighbor.
    """
    knn = KNeighborsClassifier(n_neighbors = n, weights = weight, metric = 'cosine', algorithm = 'brute')
    knn.fit(X, y)
    return knn

# Naive Bayes Classifier
def train_nb(X, y):
    """
    Create and train the Naive Baye's Classifier.
    """
    clf = MultinomialNB().fit(X, y)
    return clf

# Logistic Regression Classifier
def train_lr(X, y):
    """
    Create and train the Naive Baye's Classifier.
    """
    lr = LogisticRegression()
    lr.fit(X, y)
    return lr

#X_all=tfidf[:235]
#y_all=train_data.iloc[:,0]

#training of ensemble
def trainEnsemble(X,y):
    #dt = train_dtc(X, y)
    kn = train_knn(X, y, 3, 'distance')
    sv = train_svm(X, y)
    #nb = train_nb(X, y)
    lr = train_lr(X, y)
    #return [dt,kn,sv,nb,lr]
    return [kn,sv,lr]

def getClassifier(v_instance):
    if isinstance(v_instance,DecisionTreeClassifier):
        classifier_str='DecisionTree'
    if isinstance(v_instance,KNeighborsClassifier):
        classifier_str='KNeighborsClassifier'
    if isinstance(v_instance,SVC):
        classifier_str='SVM'
    if isinstance(v_instance,MultinomialNB):
        classifier_str='NaiveBayes'
    if isinstance(v_instance,LogisticRegression):
        classifier_str='LogisticRegression'
    return classifier_str

def predEnsemble(X,ensem):
    pred=[]
    columns=[]
    labelSet=set()
    for i in xrange(0,len(ensem)):
        print '- model',i+1
        predicted=ensem[i].predict(X)
        pred.append(predicted)
        columns.append(getClassifier(ensem[i]))
        labelSet=labelSet|set(predicted)
    ensDF=pd.DataFrame(pred).transpose()
    ensDF.columns=columns
    labelSet=list(labelSet)
    majority = ensDF.apply(pd.Series.value_counts, axis=1)[labelSet].fillna(0)
    predEnsem=majority.idxmax(axis=1)
    return predEnsem
    
#doing cross validation for ensemble  

fullIndex=range(0,235)
random.seed(42)
inSampleAcc=[]
predAcc=[]
for n in xrange(0,10):
    random.shuffle(fullIndex)
    print 'Cross Validation run: ',n+1
    for i in xrange(0,10):
        print 'k-fold: ',i+1
        #trainIndex=random.sample(xrange(235),int(math.floor(0.9*235)))
        #testIndex=list(set(xrange(235))-set(trainIndex))
        if (i==0):
            testIndex=fullIndex[:cvPartition[0]]
        else:
            testIndex=fullIndex[cvPartition[i-1]:cvPartition[i]]
        trainIndex=list(set(fullIndex)-set(testIndex))
        X_train=X_all[trainIndex]
        y_train=y_all[trainIndex]
        #train ensemble
        print 'training ensemble'
        ensem=trainEnsemble(X_train,y_train)
        #predict training
        print 'predicting train'
        predTrain=predEnsemble(X_train,ensem)
        inSampleAcc.append(accuracy_score(y_train,predTrain))
        X_test=X_all[testIndex]
        y_test=y_all[testIndex]
        #esemble prediction
        print 'predicting test'
        predTest=predEnsemble(X_test,ensem)
        predAcc.append(accuracy_score(y_test,predTest))

print 'In Sample Classification Accuracy for ensemble: ',sum(inSampleAcc)/len(inSampleAcc)
print 'Out of Sample Classification Accuracy for ensemble: ',sum(predAcc)/len(predAcc)

#In Sample Classification Accuracy for ensemble:  0.943638334973
#Out of Sample Classification Accuracy for ensemble:  0.675344202899

print tfidf.shape

#In Sample Classification Accuracy for ensemble:  0.996169856031
#Out of Sample Classification Accuracy for ensemble:  0.73981884058


#predicting for unlabelled data using feature hashing using a single SVM
svm = train_svm(X_allf, y_all)
#predicting for the unlabelled data
predUnlabelled = svm.predict(featMatrix[235:])
predUL=list(predUnlabelled)

import csv
with open('ULPredictions.csv', 'wb') as f:
    csv.writer(f).writerow(predUL)

    
#Other explorations
#Are there any clear clusters in the data set?
num_clusters = 10 #10 clusters because of 10 causes
km1 = KMeans(n_clusters=num_clusters, random_state=42)
km1.fit(tfidf)
clusters1 = km1.labels_.tolist()

array1 = np.array(clusters1)
silhouette_score(tfidf, array1, metric='euclidean', sample_size=None, random_state=None)
#0.031277350000072916


## clustering only on the training data
km2 = KMeans(n_clusters=num_clusters, random_state=42)
km2.fit(X_all)
clusters2 = km2.labels_.tolist()

array2 = np.array(clusters2)
silhouette_score(X_all, array2, metric='euclidean', sample_size=None, random_state=None)
#0.037444797109297122

from nltk.cluster import GAAClusterer
clusterer = GAAClusterer(4)
clusters_agg = clusterer.cluster(X_all.toarray(), True)
array3 = np.array(clusters_agg)
# EValuating the nltk Agglomerative clustering
silhouette_score(X_all, array3, metric='cosine', sample_size=None, random_state=None)
