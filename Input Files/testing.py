import os
import csv
import time
import pandas as pd
import numpy as np
from viterbi_decoding import viterbi

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
	print("First train the model by using the command- \"python main_training\"")
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

fn = input("Enter the name of the file where all sentences for testing are stored\n(The file should be in the Output Files folder and it should be in csv format): ")

f=open(fn,'r',encoding='utf8')
reader = csv.reader(f)

only_words=[]
given_tags=[]
predicted_tags=[]
correct_count=0
total_count=0
complete_given_tags=[]
complete_predicted_tags=[]

start=time.time()
print(start)

count=0
for line in reader:
	only_words.clear()
	given_tags.clear()
	predicted_tags.clear()
	for words in line:
		total_count+=1
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
f.close()
confusion_matrix = np.zeros((len(list_of_tags), len(list_of_tags)), dtype='float32')
for (g,p) in  zip(complete_given_tags,complete_predicted_tags):
	for(given,predicted) in zip(g,p):
		given_splitted_tags=given.split('-')
		predicted_splitted_tags=predicted.split('-')
		if any(tag in predicted_splitted_tags for tag in given_splitted_tags)==True:
			correct_count+=1
		for tag in given_splitted_tags:
			for tag2 in predicted_splitted_tags:
				confusion_matrix[list_of_tags.index(tag),list_of_tags.index(tag2)]+=1

print("Accuracy of the system is : " + str(correct_count*100/total_count) + " %")
print("Confusion Matrix added")

confusion_matrix_df = pd.DataFrame(confusion_matrix, columns = list_of_tags, index=list_of_tags)
#print(repr(confusion_matrix_df))
os.chdir(d+"/Output Files")
confusion_matrix_df.to_csv(r'confusion_matrix.csv', header=True, index=True)

only_words.clear()
given_tags.clear()
predicted_tags.clear()
complete_given_tags.clear()
complete_predicted_tags.clear()
del transition_matrix
del emission_matrix

'''import os
import csv
import time
import pandas as pd
import numpy as np
from clean_files_using_elementTree import *

d=os.path.abspath('..')
def getListOfFiles(dirName):
	listOfAllFiles = list()
	for (dirpath, dirnames, filenames) in os.walk(dirName):
		listOfAllFiles += [os.path.join(dirpath, file) for file in filenames]
	return listOfAllFiles

os.chdir(d+"/Output Files")
f=open("list_of_tags.txt",'r',encoding='utf8')
list_of_tags=[]
for line in f:
	list_of_tags.append(line.rstrip())
f.close()

f=open("transition_matrix.csv",'r',encoding='utf8')
transition_matrix_df = pd.read_csv(f,index_col='index')
f.close()

f=open("emission_matrix.csv",'r',encoding='utf8')
emission_matrix_df = pd.read_csv(f,index_col='index')
f.close()	

def Viterbi(words):
	state = []
	for key, word in enumerate(words):
		p = [] 
		for tag in list_of_tags:
			if key == 0:
				transition_p = transition_matrix_df.at['start', tag]
			else:
				transition_p = transition_matrix_df.at[state[-1], tag]

			if words[key] in emission_matrix_df.index:
				emission_p = emission_matrix_df.at[words[key], tag]
			else:
				emission_p=0
			state_probability = emission_p * transition_p   
			p.append(state_probability)
		pmax = max(p)
		state_max = list_of_tags[p.index(pmax)] 
		state.append(state_max)
	return state

#list=['THE','PLAYERS']
#print(Viterbi(list)[1])
os.chdir(d+"/Output Files")
f=open("word_tag_for_testing_file.csv",'r',encoding='utf8')
only_words=[]
given_tags=[]
predicted_tags=[]
reader = csv.reader(f)
correct_count=0
total_count=0
complete_given_tags=[]
complete_predicted_tags=[]


start = time.time()
print("Start Time: ", start)
for line in reader:
	only_words.clear()
	given_tags.clear()
	for words in line:
		total_count+=1
		only_words.append(words.split('_')[0])
		given_tags.append(words.split('_')[1])
	predicted_tags=Viterbi(only_words)
	complete_given_tags.append(given_tags)
	complete_predicted_tags.append(predicted_tags)
	for (given,predicted) in zip(given_tags,predicted_tags):
		given=given.split('-')
		if predicted in given:
			correct_count+=1
end = time.time()

difference = end-start
print("End Time: ", end) 

print("Time taken in seconds: ", difference)
print(correct_count)
print(total_count)
print("Accuracy of the system is : " + str(correctCount*100/total_count))
f.close()


# ------------------------------------------------------- #

#       read trained_file and copy into dictionary        #

# ------------------------------------------------------- #


d=os.path.abspath('..')
os.chdir(d+"/Output Files")
trained_word_tag={}
f=open("trained_file.txt",'r',encoding='utf8')
for line in f:
    line = line.replace('\n', '')
    (key, val) = line.split('_')
    trained_word_tag[key] = val
f.close()


# ------------------------------------------------------- #

#               clean files in test-corpus                #

#        and add all word_tags into seperate file         #

# ------------------------------------------------------- #


listOfTestFiles = getListOfFiles(d+"/Test-corpus")
word_tag_for_testing, testFileLen= getWordTagsFromCorpus(listOfTestFiles)                   #list of all word_tag after complete training
print("Number of word_tag's in test dataset: %d" % testFileLen)
os.chdir(d+"/Output Files")                                                                 #change the directory to create a new output file. For me it is Documents/AI_project/Output Files
f=open("word_tag_for_testing_file.txt",'w',encoding='utf8')                                 #now create a new file in current directory   
for i in sorted(word_tag_for_testing):                                                      #add all elements of list into the file
    f.write(i)
    f.write("\n")
f.close()


# ------------------------------------------------------- #

#               get accuracy of the system                #

#     get confusion matrix of the system using pandas     #

# ------------------------------------------------------- #


actualTags=[]
predictedTags=[]
correctCount=0
for item in word_tag_for_testing:
    split=item.split('_')
    actualTags.append(split[1])
    if split[0] in trained_word_tag:
        predictedTags.append(trained_word_tag[split[0]])
        if split[1]==trained_word_tag[split[0]]:
            correctCount=correctCount+1
    elif split[1]=='NN1':
        predictedTags.append('NN1')
        correctCount=correctCount+1
    else:
        predictedTags.append('NN1')
print("Accuracy of the system is : " + str(correctCount*100/testFileLen))

data = {'y_Actual':   actualTags ,
        'y_Predicted':predictedTags 
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

confusion_matrix.to_csv('confusion_matrix.csv')
f.close()


# ------------------------------------------------------- #

#          clear all used dictionarie and lists           #

# ------------------------------------------------------- #


listOfTestFiles.clear()
word_tag_for_testing.clear()
actualTags.clear()
predictedTags.clear() 
trained_word_tag.clear()'''