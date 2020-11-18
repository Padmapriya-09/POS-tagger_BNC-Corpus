import os
from pathlib import Path
from collections import Counter 
from get_list_of_files import *							            #has a function to get list of all files under a directory by taking the root directory as input. It uses os.walk()
from train_using_elementTree import * 					        #has a function which uses Element tree to train the tagger which takes list of all files for training and returns a list which contains all words along with tag in the format word_tag.
from get_freq_of_word import *							            #has a function to count frequency of all words in given list of words with tags. Returns a dictionary with word as key and frequency of word as value
from get_freq_of_word_with_tag import *					        #has a function to count frequency of a word_tag in a given list of words with tags. Returns a nested dictionary with key as word and value as another dictionary whose key is tag and value is frequency of word along with tag
from get_freq_of_tag import *							              #function to count frequency of all tags in given list of words with tags. Returns a dictionary with tag as key and frequency of tag as value


#e=os.getcwd()																	                 #get the currect working directory for me it is: Documents/AI_project/
d=os.path.abspath('..')                                          #get parent of current working directory. For me it is: Documents/AI_project
listOfTrainFiles = getListOfFiles(d+"/Train-corups")									 #get list of all files in the root directory including files from sub-directories

#to create a file and store all word_tag's after training
word_tag, trainFileLen= getWordTagsFromCorpus(listOfTrainFiles)											#list of all word_tag after complete training
print("Number of word_tag's in train dataset: %d" % trainFileLen)
os.chdir(d+"/Output Files")																	#change the directory to create a new output file. For me it is Documents/AI_project/Output Files
f=open("word_tag_file.txt",'w')														      #now create a new file in current directory	
for i in sorted(word_tag):													    #add all elements of list into the file
    f.write(i)
    f.write("\n")

#to create a file and store frequency of all words after training
#to create a list and store all available words
freq_of_word=countFrequencyOfWord(word_tag)
list_of_words=[]
print("Number of word's in train dataset : %d" % len (freq_of_word))
os.chdir(d+"/Output Files") 															#change the directory to create a new output file. For me it is Documents/AI_project/Output Files
f=open("freq_of_word_file.txt",'w')												#now create a new file in current directory	
for key, value in sorted(freq_of_word.items()):
    list_of_words.append(key)
    f.write(key)
    f.write(" : ")
    f.write(str(value))
    f.write("\n")

#to create a file and store frequency of all word_tag combinations after training
freq_of_word_with_tag=countFrequencyOfWordWithtag(word_tag)
print("Number of distinct word_tag's in train dataset : %d" % findCompleteLengthOfNestedDictionary(freq_of_word_with_tag))
os.chdir(d+"/Output Files") 
f=open("freq_of_word_tag_file.txt",'w')
for key, nested in sorted(freq_of_word_with_tag.items()):
    print(key, file=f)
    for subkey, value in sorted(nested.items()):
        print('   {}: {}'.format(subkey, value), file=f)

#to create a file and store frequency of all tags after training
#to create a list and store all available tags
freq_of_tag=countFrequencyOfTag(word_tag)
prob_of_tag={}
list_of_tags=[]
print("Number of tag's in train dataset : %d" % len (freq_of_tag))
os.chdir(d+"/Output Files") 																	#change the directory to initial for me it is: Documents/AI_project
f=open("freq_of_tag_file.txt",'w')												    #now create a new file in current directory	
for key, value in sorted(freq_of_tag.items()):
    prob_of_tag[key]=value/len(word_tag)
    list_of_tags.append(key)
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

#to create a file and store probability of a word given tag
prob_of_word_given_tag={}
os.chdir(d+"/Output Files")
f=open("prob_of_word_given_tag_file.txt",'w')
for tag in list_of_tags:
    prob_of_word_given_tag[tag]={}
    f.write(tag+":\n")
    for word in list_of_words:
        if tag in freq_of_word_with_tag[word]:
            probability_of_word_given_tag = freq_of_word_with_tag[word][tag]/freq_of_word[word]
            prob_of_word_given_tag[tag][word]=probability_of_word_given_tag
            f.write("\t"+word+":"+"\t"+str(probability_of_word_given_tag)+"\n")

prob_of_tag_given_word={}
trained_word_tag={}
os.chdir(d+"/Output Files")
f=open("trained_file.txt",'w')
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
    f.write(max_tag+"\n")
    trained_word_tag[word]=max_tag

listOfTestFiles = getListOfFiles(d+"/Test-corpus")
word_tag_for_testing, testFileLen= getWordTagsFromCorpus(listOfTestFiles)                                         #list of all word_tag after complete training
print("Number of word_tag's in test dataset: %d" % testFileLen)
os.chdir(d+"/Output Files")                                                                 #change the directory to create a new output file. For me it is Documents/AI_project/Output Files
f=open("word_tag_for_testing_file.txt",'w')                                                           #now create a new file in current directory   
for i in sorted(word_tag_for_testing):                                                      #add all elements of list into the file
    f.write(i)
    f.write("\n")

correctCount=0
for item in word_tag_for_testing:
    split=item.split('_')
    if split[0] in trained_word_tag:
        if trained_word_tag[split[0]]==split[1]:
            correctCount=correctCount+1
    elif split[1]=='NN1':
        correctCount=correctCount+1
print("Accuracy of the system is : " + str(correctCount*100/testFileLen))


#to clear all the dictionaries and lists used
freq_of_word.clear()
freq_of_tag.clear()
freq_of_word_with_tag.clear()
word_tag.clear()
list_of_words.clear()
list_of_tags.clear() 
prob_of_word_given_tag.clear()   
prob_of_tag_given_word.clear() 
word_tag_for_testing.clear() 