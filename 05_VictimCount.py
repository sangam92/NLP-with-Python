# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 23:34:46 2016

@author: yosinanggusti
"""

from nltk import word_tokenize
import csv
import numpy as np
from nltk.corpus import stopwords
from nltk import pos_tag
#Create a stopword list from the standard list of stopwords available in nltk
stop = stopwords.words('english')




#KEYWORD IDENTIFICATION 
with open('osha.csv','Ur') as csvfile:
    spamreader = csv.reader(csvfile,
                       delimiter=',')
    osha = list(spamreader)

#from collections import Counter
#counter = Counter()
mylist2 = []

for title in osha:
    #turn all to lower case
    title[1] = title[1].lower()
    #stopword removal
    title[1]=" ".join(filter(lambda word: word not in stop, title[1].split()))
    #turn all to token
    token = word_tokenize(title[1])
    token = tuple(token)
    mylist2.append(token)
set(mylist2)

#combining multiple tuples into 1 tuple
b = [i for sub in mylist2 for i in sub]
print b

#counting word freq
wordfreq = []
for w in b:
    wordfreq.append(b.count(w))

#descending order of word-freq set tells high frequency of words
a = dict(zip(b,wordfreq))
sorted_a=sorted(a.items(), key=lambda x:x[1], reverse=True)


##FIRST ITERATION - METHOD check on title column-----------------------------------------

with open('osha.csv','Ur') as csvfile:
    spamreader = csv.reader(csvfile,
                       delimiter=',')
    osha = list(spamreader)

#Create a stopword list from the standard list of stopwords available in nltk
stop = stopwords.words('english')

multiplecount = 0
singlecount=0    
for title in osha:
	#turn all to lower case
    title[1] = title[1].lower()
    title[1]=" ".join(filter(lambda word: word not in stop, title[1].split()))
    #turn all to token
    token = word_tokenize(title[1])
    #check if there are keywords contain and label accordingly
    if 'employees' in token or 'workers' in token or 'victims' in token:
        #either count multiple or label that row with multiple victom    
        multiplecount=multiplecount+1
    if 'employee' in token or 'worker' in token or 'victim' in token:
        singlecount = singlecount+1

print multiplecount
print singlecount
print singlecount+multiplecount
"""printout:
1078 -> multiple
13194 -> single
14272, this means that there are 2051 unidentified # of category; alternative is to use if this then multiple else-> single

"""

#Second iteration --- METHOD check on descriptions-----------------------------------------
#Third iteration - added plurar and singular frequent column to help identify undetected keywords
#Fourth iteration - combine the use of title and descriptions to get the single/multiple victims. 
labelledlist=[]

with open('osha.csv','Ur') as csvfile:
    spamreader = csv.reader(csvfile,
                       delimiter=',')
    osha = list(spamreader)

#Create a stopword list from the standard list of stopwords available in nltk
#stop = stopwords.words('english')

multiplecount = 0
singlecount=0    
existplural = 0
threecount=0
secndcount=0
single=0
unknown=0
threelist=[]
secndlist=[]
singlelist=[]
nounlist=[]
plurallist=['laborers', 'fighters', 'operators', 'carpenters', 'firefighters', 'electricians','drivers','supervisors','contractors','pipefitters','painters','plumbers',
'constructors','administrators','brothers','distributors','technicians','cleaners','janitors','farmers']
singularlist=['operator', 'driver', 'owner', 'electrician', 'firefighter', 'employer', 'carpenter', 'supervisor', 'laborer','painter','mechanic','foreman','contractor',
'diver','technician', 'security', 'manager','inspector','volunteer','subcontractor','plumber','passenger','pilot','cleaner','specialist']

for desc in osha:
    #turn all to lower case - removed due to pos tagging
    #desc[2] = desc[2].lower()
    #desc[3] = desc[3].lower()
    #desc[4] = desc[4].lower()
    #desc[2]=" ".join(filter(lambda word: word not in stop, desc[2].split()))
    #desc[3]=" ".join(filter(lambda word: word not in stop, desc[3].split()))
    #desc[4]=" ".join(filter(lambda word: word not in stop, desc[4].split()))
    desc[1] = desc[1].lower() #desc[1] is title column
    desc[1]=" ".join(filter(lambda word: word not in stop, desc[1].split()))
    #turn all to token
    titletoken = word_tokenize(desc[1])
    completedesc = desc[2] + desc[3] + desc[4]
    pos = pos_tag(word_tokenize(completedesc)) 
    token = word_tokenize(completedesc)
    #print pos  
    
    #check if there are keywords contain and label accordingly
    if 'employees' in titletoken or 'workers' in titletoken or 'victims' in titletoken:
        #either count multiple or label that row with multiple victom    
        multiplecount=multiplecount+1
        labelledlist.append('multiple')
    elif 'employee' in titletoken or 'worker' in titletoken or 'victim' in titletoken:
        singlecount = singlecount+1
        labelledlist.append('single')
    #only when title doesnt have enough clue then go for descriptions
    elif '#3' in completedesc:
        threecount += 1
        labelledlist.append('multiple')
    elif '#2' in completedesc:
         secndcount += 1
         labelledlist.append('multiple')
    elif '#1' in completedesc or 'employee' in completedesc or 'worker' in completedesc or 'victim' in completedesc:
        single += 1
        labelledlist.append('single')
    #else do pos tagging to those and check the noun whether it is an occupation and if can be categorised as single/plural
    #using the list of identified plural person and singular person, we try to categorise whether it is a multiple or singular victim
    #check APPENDIX 1 below for the identified words
    #THIRD ITERATION WHERE WE ADD IN PLURAL AND SINGLE NOUN CHECK
    elif any(x in token for x in plurallist):
        multiplecount += 1
        labelledlist.append('multiple')
    elif any(x in token for x in singularlist): 
        singlecount += 1
        labelledlist.append('single')
    else:#asssume single
        singlecount += 1
        labelledlist.append('single')
        
"""
    else:
        unknown += 1
        for p in pos:
            #assuming that if there are plural noun, then we will collect the plural noun, else we will append in singular list
            if 'NNS' in p[1]:
                plurallist.append(p)
            elif 'NN' in p[1]:
                singularlist.append(p)
        #print nounlist
"""
    
"""
extract out all labelled case to csv.""" 

with open('victimcount.csv', 'wb') as f:
    writer = csv.writer(f)
    for val in labelledlist:
        writer.writerow([val])


#print threecount
#print secndcount
#print single
#print unknown
#print multiplecount
#print singlecount
#print threecount+secndcount+single+unknown
print threecount+secndcount+single+ singlecount+multiplecount
print threecount+secndcount+multiplecount
print single+ singlecount

from collections import Counter
singfreqnoun = Counter(elem[0] for elem in singularlist)
print singfreqnoun.most_common

pluralfreqnoun = Counter(elem[0] for elem in plurallist)
print pluralfreqnoun.most_common

#APPENDIX 1
#identify most common 'occupation' 
#from plurallist, we manually extracted manual pluralfreqnoun:
#laborers, fighters, operators, carpenters, firefighters, electricians,drivers,supervisors,contractors,pipefitters,painters,plumbers,
#constructors,administrators,brothers,distributors,technicians,cleaners,janitors,farmers

#identified potential victim from singfreqnoun:
#operator, driver, owner, electrician, firefighter, employer, carpenter, supervisor, laborer,painter,mechanic,foreman,contractor,
#diver,technician, security, manager,inspector,volunteer,subcontractor,plumber,passenger,pilot,cleaner,specialist

"""printout from second iteration (before singfreqnoun and pluralfreqnoun):
501 -3 victims
993 -2 victims
14338 - single victim
total: 15832 , meaning 491 not identified
"""
"""iteration three where we include singfreqnoun check and pluralfreqnoun check:
501 - 3victims
993 - 2victims
14338 -1 victim
91-many victim (> 1 victims)
96-1 victim
16019 labelled - with 304 unlabelled (assume single after heuristic)
thus, ratio is 90.3% to 9.7% for single to multiple victims
"""

#plot graph
import matplotlib.pyplot as plt

xaxis = ['1', '2', '3', '>1']
frequencies = [ single+singlecount, secndcount, threecount,multiplecount]

pos = np.arange(len(xaxis))
width = 1.0     # gives histogram aspect to the bar diagram

ax = plt.axes()
ax.set_title('Number of Victim in Cases')
ax.set_ylabel('Number of cases')
ax.set_xlabel('Number of victims')
ax.set_xticks(pos + (width))
ax.set_xticklabels(xaxis)

plt.bar(pos, frequencies, width, color='b')
plt.show()

#iteration 4 result: 1255 and 15068 multiple adn single victims
#single VS multiple pie chart
labels = 'Single Victim', 'Multiple Victims'
sizes = [ 92.3, 7.7]
colors = ['gold', 'lightskyblue']
explode = (0, 0.1)  # only "explode" the 2nd slice 

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')

fig = plt.figure()


