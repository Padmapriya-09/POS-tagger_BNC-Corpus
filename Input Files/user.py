import os
import sys
import pandas as pd
from viterbi_decoding import *


# --------------------------------------------------------------------------------------------------------------------------- #

#                                      read all required inputs for viterbi from files                                        #

# --------------------------------------------------------------------------------------------------------------------------- #

d=os.path.abspath('..')
os.chdir(d+"/Output Files")
try:
	f1=open("list_of_tags.txt",'r',encoding='utf8')
	f2=open("transition_matrix.csv",'r',encoding='utf8')
	f3=open("emission_matrix.csv",'r',encoding='utf8')
	f4=open("tag_prior_probabilities.csv",'r',encoding='utf8')
	f5=open("list_of_words.txt",'r',encoding='utf8')
except OSError:
	print("Could not open/read file some files")
	print("First train the model by using the command- \"python main_training\"\nAnd select Hidden Markov Model when prompted to enter by which way you want to train the tagger")
	sys.exit()

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


# --------------------------------------------------------------------------------------------------------------------------- #

#                                               get input paragraph/sentence                                                  #

# --------------------------------------------------------------------------------------------------------------------------- #

count=0
while count<=20:
	input = input("Enter a sentence : ")
	reader=input.split(' ')

	predicted_tag_indices=viterbi(reader,TM,EM,list_of_words,list_of_tags,TPP)
	predicted_tags=[]
	for i in predicted_tag_indices:
		predicted_tags.append(list_of_tags[i])

	for item,tag in zip(reader,predicted_tags):
		print(item+'_'+tag+' ')
	count+=1
	del input