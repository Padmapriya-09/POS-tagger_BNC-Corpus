import os
from pathlib import Path
from collections import Counter 
from get_list_of_files import *							        #has a function to get list of all files under a directory by taking the root directory as input. It uses os.walk()
from clean_files_using_elementTree import * 					#has a function which uses Element tree to train the tagger which takes list of all files for training and returns a list which contains all words along with tag in the format word_tag.


# ------------------------------------------------------- #

#           function to get frequency of tag              #

# ------------------------------------------------------- #


freq_of_tag={}
def countFrequencyOfTag(my_list):
	global freq_of_tag
	for item in my_list:
		splitWords=item.split('_')
		if(splitWords[1] in freq_of_tag):
			freq_of_tag[splitWords[1]] += 1
		else:
			freq_of_tag[splitWords[1]] = 1
            

# ------------------------------------------------------- #

#            function to get frequency of word            #

# ------------------------------------------------------- #


freq_of_word = {}
def countFrequencyOfWord(my_list):  
	global freq_of_word
	for item in my_list:
		splitWords=item.split('_')
		if (splitWords[0] in freq_of_word): 
			freq_of_word[splitWords[0]] += 1
		else:
			freq_of_word[splitWords[0]] = 1
    

# ------------------------------------------------------- #

#         function to get frequency of word_tag           #

# ------------------------------------------------------- #


freq_of_word_with_tag={}
def countFrequencyOfWordWithtag(my_list):
	global freq_of_word_with_tag
	for item in my_list:
		splitWords=item.split('_')
		#print(splitWords[0]+' '+splitWords[1])
		if (splitWords[0] in freq_of_word_with_tag):							#if word is already present in dictionary then check if corresponding tag is present or not
			if(splitWords[1] in freq_of_word_with_tag[splitWords[0]]):			#if the tag is already present for the word then increase count
				freq_of_word_with_tag[splitWords[0]][splitWords[1]]+=1
			else:																#if tag is not present then just make the frequency of that tag in that word as 1
				freq_of_word_with_tag[splitWords[0]][splitWords[1]]=1
		else:																	#if word is not present in dictionary, first create a dictionary as value for the key
			freq_of_word_with_tag[splitWords[0]]={}
			freq_of_word_with_tag[splitWords[0]][splitWords[1]]=1
	return freq_of_word_with_tag


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

# ------------------------------------------------------- #


#e=os.getcwd()													#get the currect working directory for me it is: Documents/AI_project/
d=os.path.abspath('..')                                         #get parent of current working directory. For me it is: Documents/AI_project
listOfTrainFiles = getListOfFiles(d+"/Train-corups")			#get list of all files in the root directory including files from sub-directories
listOfOutputFiles = getListOfFiles(d+"/Output Files")
for f in listOfOutputFiles:
	os.remove(f)
#to create a file and store all word_tag's after training
word_tag, trainFileLen= getWordTagsFromCorpus(listOfTrainFiles)	#list of all word_tag after complete training
print("Number of word_tag's in train dataset: %d" % trainFileLen)
os.chdir(d+"/Output Files")										#change the directory to create a new output file. For me it is Documents/AI_project/Output Files
f=open("word_tag_file.txt",'w',encoding='utf8')					#now create a new file in current directory	
for i in sorted(word_tag):										#add all elements of list into the file
    f.write(i)
    f.write("\n")


# ------------------------------------------------------- #

#                   get all frequencies                   #

#            store list of all words and tags             #

#                        find P(tag)                      #

# ------------------------------------------------------- #


prob_of_tag={}
countFrequencyOfWord(word_tag)
countFrequencyOfWordWithtag(word_tag)
countFrequencyOfTag(word_tag)
list_of_words=freq_of_word.keys()
list_of_tags=freq_of_tag.keys()
print("Number of word's in train dataset : %d" % len (freq_of_word))
print("Number of distinct word_tag's in train dataset : %d" % findCompleteLengthOfNestedDictionary(freq_of_word_with_tag))
print("Number of tag's in train dataset : %d" % len (freq_of_tag))
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