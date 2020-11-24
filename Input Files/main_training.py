import os
from pathlib import Path
from collections import Counter 
from get_list_of_files import *							        #has a function to get list of all files under a directory by taking the root directory as input. It uses os.walk()
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


# ------------------------------------------------------- #

#   function to get complete length of nested dictionary  #

# ------------------------------------------------------- #


def findCompleteLengthOfNestedDictionary(dict):
	length=0
	for key,value in dict.items():
		length += len(dict[key])
	return length
    

# ------------------------------------------------------- #

#               clean files in train-corpus               #

#        and add all word_tags into seperate file         #

#                   get all frequencies                   #

# ------------------------------------------------------- #


#e=os.getcwd()													#get the currect working directory for me it is: Documents/AI_project/
d=os.path.abspath('..')                                         #get parent of current working directory. For me it is: Documents/AI_project
listOfTrainFiles = getListOfFiles(d+"/Train-corups")			#get list of all files in the root directory including files from sub-directories
listOfOutputFiles = getListOfFiles(d+"/Output Files")
for f in listOfOutputFiles:
	os.remove(f)

os.chdir(d+"/Output Files")										#change the directory to create a new output file. For me it is Documents/AI_project/Output Files
f=open("word_tag_file.txt",'w',encoding='utf8')					#now create a new file in current directory	
trainFileLen=0
totalWordTagLen=0
word_tag=[]
freq_of_tag={}
freq_of_word={}
freq_of_word_with_tag={}
for file in listOfTrainFiles:
	tree = ET.parse(file)
	root=tree.getroot()
	for sentence in root.findall('.//s'):
		line=[]
		for word in sentence.findall('.//w'):
			if word.text.strip() in freq_of_word:
				freq_of_word[word.text.strip()]+=1
			else:
				freq_of_word[word.text.strip()]=1
			trainFileLen=trainFileLen+1
			splittedTags=word.attrib['c5'].split('-')
			for tag in splittedTags:
				line.append(word.text.strip()+'_'+tag)
				#word_tag.append(word.text.strip()+'_'+tag)
				totalWordTagLen+=1
				f.write(word.text.strip()+'_'+tag+' ')
				if word.text.strip() in freq_of_word_with_tag:
					if tag in freq_of_word_with_tag[word.text.strip()]:
						freq_of_word_with_tag[word.text.strip()][tag]+=1
					else:
						freq_of_word_with_tag[word.text.strip()][tag]=1
				else:
					freq_of_word_with_tag[word.text.strip()]={}
					freq_of_word_with_tag[word.text.strip()][tag]=1
				if tag in freq_of_tag:
					freq_of_tag[tag] += 1
				else:
					freq_of_tag[tag] = 1
		if line!=[]:
			word_tag.append(line)
		f.write('\n')
print("Number of word_tag's in train dataset: %d" % trainFileLen)
print("Number of word's in train dataset : %d" % len (freq_of_word))
print("Number of distinct word_tag's in train dataset : %d" % findCompleteLengthOfNestedDictionary(freq_of_word_with_tag))
print("Number of tag's in train dataset : %d" % len (freq_of_tag))
f.close()

# ------------------------------------------------------- #

#            store list of all words and tags             #

#                        find P(tag)                      #

# ------------------------------------------------------- #


prob_of_tag={}
list_of_words=freq_of_word.keys()
list_of_tags=freq_of_tag.keys()

for key, value in sorted(freq_of_tag.items()):
    prob_of_tag[key]=value/totalWordTagLen


# ------------------------------------------------------- #

#     report top 10 frequently used words and tags        #

# ------------------------------------------------------- #


#to get the top 10 frequently used words
print('Top 10 frequently used words are:')
k = Counter(freq_of_word) 
high = k.most_common(10) 
for i in high:
	print("\t",i[0]," :",i[1]," ")

#to get the top 10 frequently used tags
print('Top 10 frequently used tags are:')
k = Counter(freq_of_tag) 
high = k.most_common(10) 
for i in high:
	print("\t",i[0]," :",i[1])
    

# ------------------------------------------------------- #

#        find emission probability = P(word|tag)          #

# ------------------------------------------------------- #


prob_of_word_given_tag={}
for tag in list_of_tags:
	prob_of_word_given_tag[tag]={}
	for word in list_of_words:
		if tag in freq_of_word_with_tag[word]:
			prob_of_word_given_tag[tag][word]=freq_of_word_with_tag[word][tag]/freq_of_tag[tag]
		else:
			prob_of_word_given_tag[tag][word]=0

# creating w x t emission matrix of words*tags, w=no.of words and t= no of tags
# Matrix(i, j) represents P(i|j)

emission_matrix = np.zeros((len(freq_of_word), len(freq_of_tag)), dtype='float32')
for i, word in enumerate(list_of_words):
	for j, tag in enumerate(list_of_tags): 
		if tag in prob_of_word_given_tag and word in prob_of_word_given_tag[tag]:
			emission_matrix[i, j] = prob_of_word_given_tag[tag][word]
 
#print(tags_matrix)

emission_matrix_df = pd.DataFrame(emission_matrix, columns = list(freq_of_tag), index=list(freq_of_word))
#print(repr(tags_df))
os.chdir(d+"/Output Files")
emission_matrix_df.to_csv(r'emission_matrix.csv', header=True, index=True)


# ------------------------------------------------------- #

#            compute transition probability               #

# ------------------------------------------------------- #

	
freq_of_tag2_given_tag1={}
start_prob={}
tag1=''
tag2=''
for line in word_tag:
	tag1=line[0].split('_')[1]
	if tag1 in start_prob:
		start_prob[tag1]+=1
	else:
		start_prob[tag1]=1
	for word in line[1:]:
		if tag1 not in freq_of_tag2_given_tag1.keys():
			freq_of_tag2_given_tag1[tag1]={}
		tag2=word.split('_')[1]
		if tag2 in freq_of_tag2_given_tag1[tag1].keys():
			freq_of_tag2_given_tag1[tag1][tag2]+=1
		else:
			freq_of_tag2_given_tag1[tag1][tag2]=1
		tag1=tag2

for tag in start_prob:
	start_prob[tag]=start_prob[tag]/freq_of_tag[tag]

print(start_prob)
# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)

transition_matrix = np.zeros((len(freq_of_tag), len(freq_of_tag)), dtype='float32')
for i, tag1 in enumerate(list_of_tags):
	for j, tag2 in enumerate(list_of_tags): 
		if tag1 in freq_of_tag2_given_tag1 and tag2 in freq_of_tag2_given_tag1[tag1]:
			transition_matrix[i, j] = freq_of_tag2_given_tag1[tag1][tag2]/freq_of_tag[tag1]

X = np.zeros((len(freq_of_tag), 1), dtype='float32')
for i,tag in enumerate(list_of_tags):
	if tag in start_prob:
		X[i] = start_prob[tag]
transition_matrix = np.hstack((X,transition_matrix))
#print(transition_matrix)

transition_matrix_columns=list(list_of_tags)
transition_matrix_columns.insert(0,'start')
transition_matrix_df = pd.DataFrame(transition_matrix, columns = transition_matrix_columns, index=list(list_of_tags))
#print(repr(tags_df))
os.chdir(d+"/Output Files")
transition_matrix_df.to_csv(r'transition_matrix.csv', header=True, index=True)


# ------------------------------------------------------- #

#  find P(tag|word) applying normalization on Bayes Rule  #

#     add most probable tags for a word into new file     #

# ------------------------------------------------------- #


prob_of_tag_given_word={}
os.chdir(d+"/Output Files")
f=open("trained_file.txt",'w',encoding='utf8')
for word in list_of_words:
    f.write(word+"_")
    maxi=0
    prob_of_tag_given_word[word]={}
    for tag in list_of_tags:
        if tag in freq_of_word_with_tag[word]:
            prob_of_tag_given_word[word][tag]=prob_of_word_given_tag[tag][word]*prob_of_tag[tag]
            if prob_of_tag_given_word[word][tag]>maxi:
                maxi=prob_of_tag_given_word[word][tag]
                max_tag=tag
    f.write(max_tag)
    f.write("\n")


# ------------------------------------------------------- #

#          clear all used dictionarie and lists           #

# ------------------------------------------------------- #


listOfTrainFiles.clear()
freq_of_word.clear()
freq_of_tag.clear()
freq_of_word_with_tag.clear()
word_tag.clear()
prob_of_word_given_tag.clear()   
prob_of_tag_given_word.clear() 