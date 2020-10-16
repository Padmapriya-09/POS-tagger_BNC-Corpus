import os
from collections import Counter 
from get_list_of_files import *							#has a function to get list of all files under a directory by taking the root directory as input. It uses os.walk()
from train_using_elementTree import * 					#has a function which uses Element tree to train the tagger which takes list of all files for training and returns a list which contains all words along with tag in the format word_tag.
from get_freq_of_word import *							#has a function to count frequency of all words in given list of words with tags. Returns a dictionary with word as key and frequency of word as value
from get_freq_of_word_with_tag import *					#has a function to count frequency of a word_tag in a given list of words with tags. Returns a nested dictionary with key as word and value as another dictionary whose key is tag and value is frequency of word along with tag
from get_complete_length_of_nested_dictionary import *	#has a function to count the complete length of nested dictionary
from get_freq_of_tag import *							#function to count frequency of all tags in given list of words with tags. Returns a dictionary with tag as key and frequency of tag as value


d=os.getcwd()																	#get the currect working directory for me it is: Documents/AI_project/
listOfFiles = getListOfFiles(d+"/Train-corups")									#get list of all files in the root directory including files from sub-directories

#to create a file and store all word_tag's after training
tagged_words= trainCorpus(listOfFiles)											#list of all word_tag after complete training
print("Number of word_tag's file after training : %d" % len (tagged_words))
os.chdir(d)																		#change the directory to initial for me it is: Documents/AI_project
f=open("word_tag.txt",'w')														#now create a new file in current directory	
for i in sorted(tagged_words):													#add all elements of list into the file
    f.write(i)
    f.write("\n")

#to create a file and store frequency of all words after training
freq_of_word=countFrequencyOfWord(tagged_words)
print("Length of frequency of word file : %d" % len (freq_of_word))
os.chdir(d)																		#change the directory to initial for me it is: Documents/AI_project
f=open("frequency_of_word.txt",'w')												#now create a new file in current directory	
for key, value in sorted(freq_of_word.items()):
   	f.write(key)
   	f.write(" : ")
   	f.write(str(value))
   	f.write("\n")

#to create a file and store frequency of all word_tag combinations after training
freq_of_word_with_tag=countFrequencyOfWordWithtag(tagged_words)
print("Length of frequency of word with tag file : %d" % findCompleteLengthOfNestedDictionary(freq_of_word_with_tag))
os.chdir(d)
f=open("frequency_of_word_with_tag.txt",'w')
for key, nested in sorted(freq_of_word_with_tag.items()):
    print(key, file=f)
    for subkey, value in sorted(nested.items()):
        print('   {}: {}'.format(subkey, value), file=f)

#to create a file and store frequency of all tags after training
freq_of_tag=countFrequencyOfTag(tagged_words)
print("Length of frequency of tag file : %d" % len (freq_of_tag))
os.chdir(d)																		#change the directory to initial for me it is: Documents/AI_project
f=open("frequency_of_tag.txt",'w')												#now create a new file in current directory	
for key, value in sorted(freq_of_tag.items()):
   	f.write(key)
   	f.write(" : ")
   	f.write(str(value))
   	f.write("\n")

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
