import os
import sys
import csv
from collections import Counter
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------------------------------------- #

#                           open file where all sentences required for training are stored                                    #

# --------------------------------------------------------------------------------------------------------------------------- #

d=os.path.abspath('..')
fn = input("Enter the name of the file with which you want to train the tagger\n(The file should be in the Output Files folder and it should be in csv format): ")
os.chdir(d+"/Output Files")
try:
    f=open(fn,'r',encoding='utf8')
except OSError:
	print("Could not open/read file: ", fn)
	print("First pre-process all the files by using the command- \"python clean_files.py\"")
	sys.exit()


# --------------------------------------------------------------------------------------------------------------------------- #

#                                 function to get complete length of nested dictionary                                        #

# --------------------------------------------------------------------------------------------------------------------------- #

def findCompleteLengthOfNestedDictionary(dict):
	length=0
	for key,value in dict.items():
		length += len(dict[key])
	return length


# --------------------------------------------------------------------------------------------------------------------------- #

#                                                   get all frequencies                                                       #

# --------------------------------------------------------------------------------------------------------------------------- #

freq_of_word={}
freq_of_tag={}
freq_of_word_with_tag={}
totalWordTagLen=0

reader = csv.reader(f)
word=''
for line in reader:
	for words in line:
		word=words.split('_')[0]
		splitted_tags=words.split('_')[1].split('-')
		totalWordTagLen+=len(splitted_tags)
		if word in freq_of_word:
			freq_of_word[word]+=1
		else:
			freq_of_word[word]=1
		if word not in freq_of_word_with_tag:
			freq_of_word_with_tag[word]={}
		for tag in splitted_tags:
			if tag in freq_of_tag:
				freq_of_tag[tag]+=1
			else:
				freq_of_tag[tag]=1
			if tag in freq_of_word_with_tag[word]:
				freq_of_word_with_tag[word][tag]+=1
			else:
				freq_of_word_with_tag[word][tag]=1
f.close()
print("Number of word's in train dataset : %d" % len (freq_of_word))
print("Number of distinct word_tag's in train dataset : %d" % findCompleteLengthOfNestedDictionary(freq_of_word_with_tag))
print("Number of tag's in train dataset : %d" % len (freq_of_tag))


# --------------------------------------------------------------------------------------------------------------------------- #

#                                  report top 10 frequently used words and tags                                               #

# --------------------------------------------------------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------------------------------------------------------- #

#                                        store list of all words and tags                                                     #

#                                      store list of tags and words in files                                                  #

#                                                    find P(tag)                                                              #

# --------------------------------------------------------------------------------------------------------------------------- #

prob_of_tag={}
list_of_words=freq_of_word.keys()
list_of_tags=freq_of_tag.keys()

os.chdir(d+"/Output Files")
f=open("list_of_tags.txt",'w',encoding='utf8')
for key in freq_of_tag:
	f.write(key+'\n')
f.close()
print("List Of Tags added")

os.chdir(d+"/Output Files")
f=open("list_of_words.txt",'w',encoding='utf8')
for key in freq_of_word:
	f.write(key+'\n')
f.close()
print("List Of Words added")

for key, value in sorted(freq_of_tag.items()):
    prob_of_tag[key]=value/totalWordTagLen


# --------------------------------------------------------------------------------------------------------------------------- #

#       			                     P(word|tag)	[required for both models]                            			      #

# --------------------------------------------------------------------------------------------------------------------------- #


prob_of_word_given_tag={}
for tag in list_of_tags:
	prob_of_word_given_tag[tag]={}
	for word in list_of_words:
		if tag in freq_of_word_with_tag[word]:
			prob_of_word_given_tag[tag][word]=freq_of_word_with_tag[word][tag]/freq_of_tag[tag]
		else:
			prob_of_word_given_tag[tag][word]=0


# --------------------------------------------------------------------------------------------------------------------------- #

#                                                using bayes_rule for training                                                #

# --------------------------------------------------------------------------------------------------------------------------- #

prob_of_tag_given_word={}
def bayes_rule():

	# ----------------------------------------------------------------------------- #

	#      delete all previously created files by this model if they exist          #

	# ----------------------------------------------------------------------------- #
	
	os.chdir(d+"/Output Files")
	if os.path.exists('trained_file_using_bayes_rule.txt'):
		os.remove('trained_file_using_bayes_rule.txt')
	
	# ----------------------------------------------------------------------------- #

	#             find P(tag|word) applying normalization on Bayes Rule             #

	#                 add most probable tags for a word into new file               #

	# ----------------------------------------------------------------------------- #

	global prob_of_tag_given_word
	os.chdir(d+"/Output Files")
	f=open("trained_file_using_bayes_rule.txt",'w',encoding='utf8')
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
	f.close()


# --------------------------------------------------------------------------------------------------------------------------- #

#                                                using hmm model for training                                                 #

# --------------------------------------------------------------------------------------------------------------------------- #

emission_matrix = np.zeros((len(list_of_tags), len(list_of_words)+1), dtype='float32')
emission_matrix_df = pd.DataFrame(emission_matrix, columns = list(list_of_words)+['<NEW WORD>'], index=list_of_tags)
freq_of_tag2_given_tag1={}
outgoing_tag_count={}
transition_matrix = np.zeros((len(list_of_tags), len(list_of_tags)), dtype='float32')
transition_matrix_df = pd.DataFrame(transition_matrix, index = list_of_tags, columns=list_of_tags)
tag_prior_probabilities = np.zeros((len(list_of_tags)), dtype='float32')
tag_prior_probabilities_df = pd.DataFrame(tag_prior_probabilities, index = list_of_tags, columns=['P(tag)'])

def hmm():
	
	# ----------------------------------------------------------------------------- #

	#      delete all previously created files by this model if they exist          #

	# ----------------------------------------------------------------------------- #
	
	os.chdir(d+"/Output Files")
	if os.path.exists('emission_matrix.csv'):
		os.remove('emission_matrix.csv')
	if os.path.exists('transition_matrix.csv'):
		os.remove('transition_matrix.csv')
	if os.path.exists('tag_prior_probabilities.csv'):
		os.remove('tag_prior_probabilities.csv')

	# ----------------------------------------------------------------------------- #

	#                 compute emission probability = P(word|tag)                    #
	
	# creating t x w emission matrix of tags*words, w=no.of words and t= no of tags #
	
	#                        Matrix(i, j) represents P(j|i)                         #

	# ----------------------------------------------------------------------------- #

	global emission_matrix
	for i, tag in enumerate(list_of_tags):
		for j, word in enumerate(list_of_words): 
			if tag in prob_of_word_given_tag and word in prob_of_word_given_tag[tag]:
				emission_matrix[i, j] = prob_of_word_given_tag[tag][word]
	emission_matrix[:,-1]=1e-12

	global emission_matrix_df
	#print(repr(emission_matrix_df))
	os.chdir(d+"/Output Files")
	emission_matrix_df.to_csv(r'emission_matrix.csv', header=True, index=True, index_label='index')
	print("Emission Matrix added")
	
	# ----------------------------------------------------------------------------- #

	#                compute transition probability = P(tag2|tag1)                  #

	# ----------------------------------------------------------------------------- #
		
	global freq_of_tag2_given_tag1
	global outgoing_tag_count
	os.chdir(d+"/Output Files")
	try:
		f=open(fn,'r',encoding='utf8')
	except OSError:
		print("Could not open/read file: ", fn)
		print("First pre-process all the files by using the command- \"python clean_files.py\"")
		sys.exit()
	reader = csv.reader(f)	
			
	for line in reader:
		tag=['start']
		for words in line:
			splitted_tags=words.split('_')[1].split('-')
			for tag1 in tag:
				if tag1 not in freq_of_tag2_given_tag1:
					freq_of_tag2_given_tag1[tag1]={}
				for tag2 in splitted_tags:
					if tag2 in freq_of_tag2_given_tag1[tag1]:
						freq_of_tag2_given_tag1[tag1][tag2]+=1
					else:
						freq_of_tag2_given_tag1[tag1][tag2]=1
					if tag1 in outgoing_tag_count:
						outgoing_tag_count[tag1]+=1
					else:
						outgoing_tag_count[tag1]=1
			tag=splitted_tags

	# ----------------------------------------------------------------------------- #
	
	#         creating t x t transition matrix of tags*tags, t= no of tags          #
	
	#                        Matrix(i, j) represents P(j|i)                         #

	# ----------------------------------------------------------------------------- #

	global transition_matrix
	for i, tag1 in enumerate(list_of_tags):
		for j, tag2 in enumerate(list_of_tags): 
			if tag1 in freq_of_tag2_given_tag1 and tag2 in freq_of_tag2_given_tag1[tag1]:
				transition_matrix[i, j] = freq_of_tag2_given_tag1[tag1][tag2]/outgoing_tag_count[tag1]

	global transition_matrix_df
	#print(repr(transitin_matrix_df))
	os.chdir(d+"/Output Files")
	transition_matrix_df.to_csv(r'transition_matrix.csv', header=True, index=True, index_label='index')
	print("Transition Matrix added")
	
	# ----------------------------------------------------------------------------- #

	#                          create tag prior probabilities                       #

	# ----------------------------------------------------------------------------- #

	global tag_prior_probabilities
	for i, tag in enumerate(freq_of_tag.keys()):
		tag_prior_probabilities[i]=freq_of_tag[tag]/totalWordTagLen

	global tag_prior_probabilities_df
	#print(repr(transitin_matrix_df))
	os.chdir(d+"/Output Files")
	tag_prior_probabilities_df.to_csv(r'tag_prior_probabilities.csv',header=True, index=True, index_label='tag')
	print("Tag Prior Probabilities added")


# --------------------------------------------------------------------------------------------------------------------------- #

#                                          switch-case statements to choose model                                             #

# --------------------------------------------------------------------------------------------------------------------------- #

boolean=False
while boolean==False:
	choice=input("Choose by which method you want to train the POS-tagger: \n\t1.Using Bayes Rule\n\t2.Using Hidden Markov Model\nYour choice!??: ")
	if choice=='1':
		boolean=True
		bayes_rule()
		break
	elif choice=='2':
		boolean==True
		hmm()
		break
	else:
		print("Choose either 1 or 2")


# --------------------------------------------------------------------------------------------------------------------------- #

#                                             clear all used dictionaries and lists                                           #

# --------------------------------------------------------------------------------------------------------------------------- #

freq_of_word.clear()
freq_of_tag.clear()
freq_of_word_with_tag.clear()
prob_of_word_given_tag.clear()   
prob_of_tag_given_word.clear() 
freq_of_tag2_given_tag1.clear()
outgoing_tag_count.clear()
del emission_matrix_df
del transition_matrix_df
del tag_prior_probabilities_df