import pandas as pd
from main_training import trained_word_tag

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
        if split[1]==trained_word_tag[split[0]]:
            correctCount=correctCount+1
    elif split[1]=='NN1':
        correctCount=correctCount+1
print("Accuracy of the system is : " + str(correctCount*100/testFileLen))

actualTags=[]
for item in word_tag_for_testing:
    split=item.split('_')
    actualTags.append(split[1])
predictedTags=[]
for item in word_tag_for_testing:
    split=item.split('_')
    if split[0] in trained_word_tag:
        predictedTags.append(trained_word_tag[word])
    else:
        predictedTags.append('NN1')
print(actualTags)
print(predictedTags)
data = {'y_Actual':   actualTags ,
        'y_Predicted':predictedTags 
        }

df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])

confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print (confusion_matrix)

listOfTestFiles.clear()
word_tag_for_testing.clear() 
