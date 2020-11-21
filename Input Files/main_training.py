import os
from pathlib import Path
from collections import Counter 
from get_list_of_files import *							        #has a function to get list of all files under a directory by taking the root directory as input. It uses os.walk()
import xml.etree.ElementTree as ET


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
substring="-"
word_tag=[]
freq_of_tag={}
freq_of_word={}
freq_of_word_with_tag={}
for file in listOfTrainFiles:
    tree = ET.parse(file)
    root=tree.getroot()
    for sentence in root.findall('.//s'):
        for word in sentence.findall('.//w'):
            trainFileLen=trainFileLen+1
            if (word.text.strip() in freq_of_word): 
                freq_of_word[word.text.strip()] += 1
            else:
                freq_of_word[word.text.strip()] = 1
            if substring in word.attrib['c5']:
                splittedTags=word.attrib['c5'].split('-')
                for tag in splittedTags:
                    word_tag.append(word.text.strip()+'_'+tag)
                    f.write(word.text.strip()+'_'+tag+'\n')
                    if (word.text.strip() in freq_of_word_with_tag):
                        if(tag in freq_of_word_with_tag[word.text.strip()]):
                            freq_of_word_with_tag[word.text.strip()][tag]+=1
                        else:
                            freq_of_word_with_tag[word.text.strip()][tag]=1
                    else:
                        freq_of_word_with_tag[word.text.strip()]={}
                        freq_of_word_with_tag[word.text.strip()][tag]=1
                    if(tag in freq_of_tag):
                        freq_of_tag[tag] += 1
                    else:
                        freq_of_tag[tag] = 1
            else:
                word_tag.append(word.text.strip()+'_'+word.attrib['c5'])
                f.write(word.text.strip()+'_'+word.attrib['c5']+'\n')
                if word.text.strip() in freq_of_word_with_tag:
                    if word.attrib['c5'] in freq_of_word_with_tag[word.text.strip()]:
                        freq_of_word_with_tag[word.text.strip()][word.attrib['c5']]+=1
                    else:
                        freq_of_word_with_tag[word.text.strip()][word.attrib['c5']]=1
                else:
                    freq_of_word_with_tag[word.text.strip()]={}
                    freq_of_word_with_tag[word.text.strip()][word.attrib['c5']]=1
                if(word.attrib['c5'] in freq_of_tag):
                    freq_of_tag[word.attrib['c5']] += 1
                else:
                    freq_of_tag[word.attrib['c5']] = 1
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
    prob_of_tag[key]=value/len(word_tag)


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

#                     find P(word|tag)                    #

# ------------------------------------------------------- #


prob_of_word_given_tag={}
for tag in list_of_tags:
    prob_of_word_given_tag[tag]={}
    for word in list_of_words:
        if tag in freq_of_word_with_tag[word]:
            probability_of_word_given_tag = freq_of_word_with_tag[word][tag]/freq_of_word[word]
            prob_of_word_given_tag[tag][word]=probability_of_word_given_tag
        else:
            prob_of_word_given_tag[tag][word]=0


# ------------------------------------------------------- #

#               Viterbi Algorithm                         #

# ------------------------------------------------------- #





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