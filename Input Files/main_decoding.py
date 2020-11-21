import os
import pandas as pd
from get_list_of_files import *
from clean_files_using_elementTree import *


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

f=open('confusion_matrix.txt','w',encoding='utf8')
f.write(str(confusion_matrix))
f.close()


# ------------------------------------------------------- #

#          clear all used dictionarie and lists           #

# ------------------------------------------------------- #


listOfTestFiles.clear()
word_tag_for_testing.clear()
actualTags.clear()
predictedTags.clear() 
trained_word_tag.clear()