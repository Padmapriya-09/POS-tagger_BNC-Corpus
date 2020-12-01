import os
import csv
import time
import sys
import pandas as pd
import numpy as np
from viterbi_decoding import viterbi


# --------------------------------------------------------------------------------------------------------------------------- #

#                           open file where all sentences required for testing are stored                                     #

# --------------------------------------------------------------------------------------------------------------------------- #

d=os.path.abspath('..')
fn = input("Enter the name of the file with which you want to test the tagger\n(The file should be in the Output Files folder and it should be in csv format): ")
os.chdir(d+"/Output Files")
try:
    f=open(fn,'r',encoding='utf8')
except OSError:
	print("Could not open/read file: ", fn)
	print("First pre-process all the files by using the command- \"python clean_files.py\"")
	sys.exit()


# --------------------------------------------------------------------------------------------------------------------------- #

#                           delete all previously created files by this program if they exist                                 #

# --------------------------------------------------------------------------------------------------------------------------- #

os.chdir(d+"/Output Files")
if os.path.exists('confusion_matrix.csv'):
	os.remove('confusion_matrix.csv')


# --------------------------------------------------------------------------------------------------------------------------- #

#                                       initialize lists required by both the models                                          #

# --------------------------------------------------------------------------------------------------------------------------- #

reader = csv.reader(f)

f1=open("list_of_tags.txt",'r',encoding='utf8')
f5=open("list_of_words.txt",'r',encoding='utf8')
list_of_tags=[]
for line in f1:
	list_of_tags.append(line.rstrip())
f1.close()
print('Completed reading the list of tags')

list_of_words=[]
for line in f5:
	list_of_words.append(line.rstrip())
f5.close()
print('Completed reading list of words')

only_words=[]
given_tags=[]
predicted_tags=[]
complete_given_tags=[]
complete_predicted_tags=[]


# --------------------------------------------------------------------------------------------------------------------------- #

#                                                using bayes_rule for testing                                                 #

# --------------------------------------------------------------------------------------------------------------------------- #

trained_word_tag={}
def bayes_rule():
	
	# ----------------------------------------------------------------------------- #

	#                       check and open required input files                     #

	# ----------------------------------------------------------------------------- #
	
	os.chdir(d+"/Output Files")
	try:
		f1=open("trained_file_using_bayes_rule.txt",'r',encoding='utf8')
	except OSError:
		print("Could not open/read file trained_file_using_bayes_rule.txt",)
		print("First train the model by using the command- \"python training\"\nAnd select Bayes rule when prompted to enter by which way you want to train the tagger")
		sys.exit()
	
	# ----------------------------------------------------------------------------- #

	#                 read trained_file and copy into dictionary                    #

	# ----------------------------------------------------------------------------- #
	
	global trained_word_tag
	for line in f1:
		line = line.replace('\n', '')
		(key, val) = line.split('_')
		trained_word_tag[key] = val
	f1.close()
	
	# ----------------------------------------------------------------------------- #

	#                         get all given and predicted tags                      #

	# ----------------------------------------------------------------------------- #
	
	complete_given_tags.clear()
	complete_predicted_tags.clear()
	for line in reader:
		given_tags.clear()
		predicted_tags.clear()
		for words in line:
			split=words.split('_')
			given_tags.append(split[1])
			if split[0] in trained_word_tag:
				predicted_tags.append(trained_word_tag[split[0]])
			else:
				predicted_tags.append('NN1')
		complete_given_tags.append(given_tags[:])
		complete_predicted_tags.append(predicted_tags[:])


# --------------------------------------------------------------------------------------------------------------------------- #

#                                                using hmm model for testing                                                  #

# --------------------------------------------------------------------------------------------------------------------------- #

def hmm():

	# ----------------------------------------------------------------------------- #

	#                       check and open required input files                     #

	# ----------------------------------------------------------------------------- #
	
	os.chdir(d+"/Output Files")
	try:
		f2=open("transition_matrix.csv",'r',encoding='utf8')
		f3=open("emission_matrix.csv",'r',encoding='utf8')
		f4=open("tag_prior_probabilities.csv",'r',encoding='utf8')
	except OSError:
		print("Could not open/read file some files")
		print("First train the model by using the command- \"python main_training\"\nAnd select Hidden Markov Model when prompted to enter by which way you want to train the tagger")
		sys.exit()
	
	# ----------------------------------------------------------------------------- #

	#               read all required inputs for viterbi from files                 #

	# ----------------------------------------------------------------------------- #

	emission_matrix = pd.read_csv(f3,index_col='index',header=0)
	EM=emission_matrix.to_numpy()
	f3.close()
	print('Completed reading emission matrix')

	transition_matrix = pd.read_csv(f2,index_col='index',header=0)
	TM=transition_matrix.to_numpy()
	f2.close()
	print('Completed reading transition matrix')	

	tag_prior_probabilities=pd.read_csv(f4,index_col='tag')
	TPP=tag_prior_probabilities['P(tag)'].values.tolist()
	f4.close()
	print('Completed reading tag prior probabilities')
	
	start=time.time()
	print(start)
	
	# ----------------------------------------------------------------------------- #

	#                         get all given and predicted tags                      #

	# ----------------------------------------------------------------------------- #
	
	complete_predicted_tags.clear()
	complete_given_tags.clear()
	count=0
	for line in reader:
		only_words.clear()
		given_tags.clear()
		predicted_tags.clear()
		for words in line:
			only_words.append(words.split('_')[0])
			given_tags.append(words.split('_')[1])
		predicted_tag_indices=viterbi(only_words,TM,EM,list_of_words,list_of_tags,TPP)
		for i in predicted_tag_indices:
			predicted_tags.append(list_of_tags[i])
		complete_given_tags.append(given_tags[:])
		complete_predicted_tags.append(predicted_tags[:])
		count+=1
		print(count)
	
	end=time.time()
	print(end)
	print("Time Taken: "+str(end-start))
	
	del emission_matrix
	del transition_matrix
	del tag_prior_probabilities


# --------------------------------------------------------------------------------------------------------------------------- #

#                                          switch-case statements to choose model                                             #

# --------------------------------------------------------------------------------------------------------------------------- #

boolean=False
while boolean==False:
	choice=input("Choose by which method you want to test the POS-tagger: \n\t1.Using Bayes Rule\n\t2.Using Hidden Markov Model\nYour choice!??: ")
	if choice=='1':
		bool=True
		bayes_rule()
		break
	elif choice=='2':
		bool==True
		hmm()
		break
	else:
		print("Choose either 1 or 2")

f.close()


# --------------------------------------------------------------------------------------------------------------------------- #

#                                            get accuracy and confusion matrix                                                #

# --------------------------------------------------------------------------------------------------------------------------- #

correct_count=0
total_count=0
confusion_matrix = np.zeros((len(list_of_tags), len(list_of_tags)), dtype='int32')
for (g,p) in  zip(complete_given_tags,complete_predicted_tags):
	for(given,predicted) in zip(g,p):
		total_count+=1
		given_splitted_tags=given.split('-')
		predicted_splitted_tags=predicted.split('-')
		if any(tag in predicted_splitted_tags for tag in given_splitted_tags)==True:
			correct_count+=1
		for tag in given_splitted_tags:
			for tag2 in predicted_splitted_tags:
				confusion_matrix[list_of_tags.index(tag),list_of_tags.index(tag2)]+=1

print("Accuracy of the system is : " + str(correct_count*100/total_count) + " %")

confusion_matrix_df = pd.DataFrame(confusion_matrix, columns = list_of_tags, index=list_of_tags)
#print(repr(confusion_matrix_df))
os.chdir(d+"/Output Files")
confusion_matrix_df.to_csv(r'confusion_matrix.csv', header=True, index=True)

print("Confusion Matrix added")

# --------------------------------------------------------------------------------------------------------------------------- #

#                                          clear all used dictionaries and lists                                              #

# --------------------------------------------------------------------------------------------------------------------------- #

list_of_tags.clear()
list_of_words.clear()
trained_word_tag.clear()
only_words.clear()
given_tags.clear()
predicted_tags.clear()
complete_given_tags.clear()
complete_predicted_tags.clear()
del confusion_matrix_df