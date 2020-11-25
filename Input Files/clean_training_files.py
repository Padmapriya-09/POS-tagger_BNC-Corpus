import os
import csv
import xml.etree.ElementTree as ET

# ------------------------------------------------------- #

#    function to get list of all files in a directory     #

# ------------------------------------------------------- #

def getListOfFiles(dirName):
	listOfAllFiles = list()
	for (dirpath, dirnames, filenames) in os.walk(dirName):
		listOfAllFiles += [os.path.join(dirpath, file) for file in filenames]
	return listOfAllFiles


# ------------------------------------------------------- #

#               clean files in train-corpus               #

#        and add all word_tags into seperate file         #

#                   get all frequencies                   #

# ------------------------------------------------------- #

d=os.path.abspath('..')                                         #get parent of current working directory. For me it is: Documents/AI_project
listOfTrainFiles = getListOfFiles(d+"/Train-corups")			#get list of all files in the root directory including files from sub-directories
listOfOutputFiles = getListOfFiles(d+"/Output Files")
for f in listOfOutputFiles:
	os.remove(f)

os.chdir(d+"/Output Files")										#change the directory to create a new output file. For me it is Documents/AI_project/Output Files
f=open("word_tag_file.csv",'w',encoding='utf8',newline ='')
writer = csv.writer(f)
line=[]
trainFileLen=0
for file in listOfTrainFiles:
	tree = ET.parse(file)
	root=tree.getroot()
	for sentence in root.findall('.//s'):
		line.clear()
		for word in sentence.findall('.//w'):
			trainFileLen+=1
			splittedTags=word.attrib['c5'].split('-')
			for tag in splittedTags:
				line.append(word.text.strip()+'_'+tag)
		if line!=[]:
			writer.writerow(line)


print("Number of word_tag's in train dataset: %d" % trainFileLen)
f.close()

listOfTrainFiles.clear()
listOfOutputFiles.clear()