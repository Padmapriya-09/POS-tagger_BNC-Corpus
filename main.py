import os
import xml.etree.ElementTree as ET
from collections import Counter 

def getListOfFiles(dirName):													#gets list of all files in directory tree for the given root
	listOfAllFiles = list()
	for (dirpath, dirnames, filenames) in os.walk(dirName):
		listOfAllFiles += [os.path.join(dirpath, file) for file in filenames]
	return listOfAllFiles

def trainCorpus(listOfFiles):
	tagged_words=[]
	for f in listOfFiles:	
		train_file = open(f)
		tree = ET.parse(train_file)
		root=tree.getroot()
		for word in root.findall('.//w'):										#finds only elements with a tag which are direct children of the current element
			tagged_words.append(word.text.strip()+"_"+word.attrib['pos'])

	return tagged_words

#freq_of_word_with_tag = {}

def countFrequencyOfWord(my_list):  
	freq_of_word = {}
	for item in my_list:
		splitWords=item.split('_')
		if (splitWords[0] in freq_of_word): 
			freq_of_word[splitWords[0]] += 1
		else:
			freq_of_word[splitWords[0]] = 1
	return freq_of_word

def countFrequencyOfWordWithtag(my_list):
	freq_of_word_with_tag={}
	for item in my_list:
		splitWords=item.split('_')
		#print(splitWords[0]+' '+splitWords[1])
		if (splitWords[0] in freq_of_word_with_tag):
			if(splitWords[1] in freq_of_word_with_tag[splitWords[0]]):
				freq_of_word_with_tag[splitWords[0]][splitWords[1]]+=1
			else:
				freq_of_word_with_tag[splitWords[0]][splitWords[1]]=1
		else:
			freq_of_word_with_tag[splitWords[0]]={}
			freq_of_word_with_tag[splitWords[0]][splitWords[1]]=1
	return freq_of_word_with_tag

def findCompleteLengthOfNestedDictionary(dict):
	length=0
	for key,value in dict.items():
		length += len(dict[key])
	return length

def countFrequencyOfTag(my_list):
	freq_of_tag={}
	for item in my_list:
		splitWords=item.split('_')
		if(splitWords[1] in freq_of_tag):
			freq_of_tag[splitWords[1]] += 1
		else:
			freq_of_tag[splitWords[1]] = 1
	return freq_of_tag

d=os.getcwd()																	#get the currect working directory here it is: Downloads/AI_project/
listOfFiles = getListOfFiles(d+"/Train-corups")									#get list of all files in the root directory including files from sub-directories

tagged_words= trainCorpus(listOfFiles)											#list of all word_tag after complete training
print("Number of word_tag's file after training : %d" % len (tagged_words))
os.chdir(d)																		#change the directory to initial here it is: Downloads/AI_project
f=open("word_tag.txt",'w')														#now create a new file in current directory	
for i in sorted(tagged_words):															#add all elements of list into the file
    f.write(i)
    f.write("\n")

freq_of_word=countFrequencyOfWord(tagged_words)
print("Length of frequency of word file : %d" % len (freq_of_word))
os.chdir(d)
f=open("frequency_of_word.txt",'w')
for key, value in sorted(freq_of_word.items()):
   	f.write(key)
   	f.write(" : ")
   	f.write(str(value))
   	f.write("\n")

freq_of_word_with_tag=countFrequencyOfWordWithtag(tagged_words)
print("Length of frequency of word with tag file : %d" % findCompleteLengthOfNestedDictionary(freq_of_word_with_tag))
os.chdir(d)
f=open("frequency_of_word_with_tag.txt",'w')
for key, nested in sorted(freq_of_word_with_tag.items()):
    print(key, file=f)
    for subkey, value in sorted(nested.items()):
        print('   {}: {}'.format(subkey, value), file=f)

print('Top 10 frequently used words are:')
k = Counter(freq_of_word) 
high = k.most_common(10) 
for i in high:
	print("\t",i[0]," :",i[1]," ")

freq_of_tag=countFrequencyOfTag(tagged_words)
print('Top 10 frequently used tags are:')
k = Counter(freq_of_tag) 
high = k.most_common(10) 
for i in high:
	print("\t",i[0]," :",i[1])