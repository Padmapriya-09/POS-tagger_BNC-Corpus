import os
import sys
import csv
from collections import Counter 
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


d=os.path.abspath('..')

# ------------------------------------------------------- #

#   function to get complete length of nested dictionary  #

# ------------------------------------------------------- #

def findCompleteLengthOfNestedDictionary(dict):
	length=0
	for key,value in dict.items():
		length += len(dict[key])
	return length


freq_of_word={}
freq_of_tag={}
freq_of_word_with_tag={}
totalWordTagLen=0

os.chdir(d+"/Output Files")
try:
    f=open("word_tag_file.csv",'r',encoding='utf8')
except OSError:
	print("Could not open/read file: ", "word_tag_file.csv")
	print("First pre-process all the training file by using the command- \"python clean_training_files.py\"")
	sys.exit()
f=open("word_tag_file.csv",'r',encoding='utf8')
reader = csv.reader(f)
word=''
tag=''
for line in reader:
	for words in line:
		totalWordTagLen+=1
		word=words.split('_')[0]
		tag=words.split('_')[1]
		if word in freq_of_word:
			freq_of_word[word]+=1
		else:
			freq_of_word[word]=1
		if word in freq_of_word_with_tag:
			if tag in freq_of_word_with_tag[word]:
				freq_of_word_with_tag[word][tag]+=1
			else:
				freq_of_word_with_tag[word][tag]=1
		else:
			freq_of_word_with_tag[word]={}
			freq_of_word_with_tag[word][tag]=1
		if tag in freq_of_tag:
			freq_of_tag[tag] += 1
		else:
			freq_of_tag[tag] = 1
f.close()
print("Number of word's in train dataset : %d" % len (freq_of_word))
print("Number of distinct word_tag's in train dataset : %d" % findCompleteLengthOfNestedDictionary(freq_of_word_with_tag))
print("Number of tag's in train dataset : %d" % len (freq_of_tag))


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

#            store list of all words and tags             #

#                        find P(tag)                      #

# ------------------------------------------------------- #

prob_of_tag={}
list_of_words=freq_of_word.keys()
list_of_tags=freq_of_tag.keys()

for key, value in sorted(freq_of_tag.items()):
    prob_of_tag[key]=value/totalWordTagLen


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
outgoing_tag_count={}
os.chdir(d+"/Output Files")
f=open("word_tag_file.csv",'r',encoding='utf8')
reader = csv.reader(f)
for line in reader:
	tag1='start'
	tag2=''
	for words in line:
		tag2=words.split('_')[1]
		if tag1 not in freq_of_tag2_given_tag1:
			freq_of_tag2_given_tag1[tag1]={}
		if tag2 in freq_of_tag2_given_tag1[tag1]:
			freq_of_tag2_given_tag1[tag1][tag2]+=1
		else:
			freq_of_tag2_given_tag1[tag1][tag2]=1
		if tag1 in outgoing_tag_count:
			outgoing_tag_count[tag1] += 1
		else:
			outgoing_tag_count[tag1] = 1
		tag1 = tag2

# creating t x t transition matrix of tags, t= no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)

list_tags=list(list_of_tags)
list_tags.insert(0,'start')
transition_matrix = np.zeros((len(list_tags), len(list_of_tags)), dtype='float32')
for i, tag1 in enumerate(list_tags):
	for j, tag2 in enumerate(list_of_tags): 
		if tag1 in freq_of_tag2_given_tag1 and tag2 in freq_of_tag2_given_tag1[tag1]:
			transition_matrix[i, j] = freq_of_tag2_given_tag1[tag1][tag2]/outgoing_tag_count[tag1]

transition_matrix_df = pd.DataFrame(transition_matrix, index = list_tags, columns=list_of_tags)
#print(repr(transitin_matrix_df))
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

#          clear all used dictionaries and lists          #

# ------------------------------------------------------- #


freq_of_word.clear()
freq_of_tag.clear()
freq_of_word_with_tag.clear()
prob_of_word_given_tag.clear()   
prob_of_tag_given_word.clear() 
list_tags.clear()