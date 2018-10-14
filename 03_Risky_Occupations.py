# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 18:23:09 2016

@author: MY PC
"""

import nltk
import os
import pandas as pd
import re
from nltk import *
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import stem
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import treebank
import csv
import itertools
import collections
from collections import Counter

## Following lines code is importing the data in a list named as description.

folder_path = 'D:\\Semester 2\\Electives\\Text Mining\\CA workout\\osha.csv'

with open(folder_path,"rb") as f:
    reader = csv.reader(f,delimiter=',')
    data = list(reader)

data1 = pd.DataFrame(data)   
description = list(data1[1])

#### Following lines of code is the declaration of the regular 
###  expressions used in the feature extraction.

empGrammer = re.compile(r"[^employee]")


grammar = r"""
  NP: {<NN>+}
      {<NNS>}
"""

cp = nltk.RegexpParser(grammar)
i = 0
colOneList = []  ##ist creation to hold the data.
indexList = []  ## list creation to hold the indexes.

for d in description:
    # Convert all characters to Lower case
    text_lower=d.lower()
    token = nltk.word_tokenize(text_lower)
    pos_tagged_preProcess = nltk.pos_tag(token)
    nouns = cp.parse(pos_tagged_preProcess)
    for subtree in nouns.subtrees(filter=lambda t: t.label() == 'NP'):
        npCnt = 0
        npCnt = npCnt + 1
        i = i+1
        if(npCnt == 1):
           colOneList.append(' '.join(term for term,postype in subtree.leaves()))
           indexList.append(i)
        break
        
        
print len(colOneList) ## colOneList holds the first noun from the title.
print len(indexList) ## holding the indexes.
        
listwoEmp = []  ## this list will hold the risky occupations other then 'Employee' and 'Worker'
updatedIndexList = [] ## the index list for which we get occupation other then 'employee' and 'worker'
updatedIndexList = indexList
print len(updatedIndexList)
j = 0

## Following lines of code is variable declaration to be used in if condition to 
## remove irrelavant noun-noun combination from the list and give the list for
## occupations.

e = "employee"
w = "worker"
es = "employees"
ws = "workers"
s = "shock"
burn = "burn"
fall = "fall"
empStrk =  "employee struck"
expln = "explosion"
truck = "truck"
empCgt = "employee caught"
empFinger = "employee finger"
crane = "crane"
caught = "caught"
fingers = "fingers"
dies = "dies"
smkInh = "smoke inhalation"
trc = "trench"
wtr = "water"
vap = "vapors"
efoot = "employee foot"
whl = "wheel"
drn = "drowns"
stk = "strikes"
empArm = "employee arm"
empSfr = "employee suffer"
wal = "wall"
fumes = "fumes"
skid = "skid"
gas = "gases"
propane = "propane"
for d in colOneList:
    j = j+1
    if(d != e) and (d != es) and (d != s) and (d != w) and (d != ws) and (d != burn) and (d != fall) and (d != empStrk) and (d != expln) and (d != truck) and (d != empCgt) and (d != empFinger) and (d != crane) and (d != caught) and (d != fingers) and (d != dies) and (d != smkInh) and (d != trc) and (d != wtr) and (d != vap) and (d != efoot) and (d != whl) and (d != drn) and (d!= stk) and (d != empArm) and (d!= empSfr) and (d != wal) and (d != fumes) and (d != skid) and (d != gas) and (d != propane):
        print "Yeh baat"
        listwoEmp.append(d)
    else: 
        print "koi baat nhi"
        updatedIndexList.remove(j)

### 'listwoEmp' is the list occupations formed by traversing title i.e 1st column of data.
        

print len(Counter(listwoEmp))
print len(listwoEmp)
print listwoEmp
print Counter(listwoEmp)

rr = open("occupationList.csv","w+")
rr.write(str(listwoEmp))
# Read the whole text.
# Generate a word cloud image
def create_cloud(text_to_draw,filename):
    wordcloud = WordCloud().generate(text_to_draw)
    # Display the generated image:
    # the matplotlib way:
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud)
    plt.axis("off")
    
    # take relative word frequencies into account, lower max_font_size
    wordcloud = WordCloud(max_font_size=40 ,relative_scaling=.5).generate(text_to_draw)
    wordcloud.to_file(path.join(d, filename))
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

## manualy editing the csv created in above code named 'occupationList.csv' to 'ocpnList.csv' 
occupations = pd.read_csv('ocpnList.csv')
create_cloud(str(occupations),"fua.png")