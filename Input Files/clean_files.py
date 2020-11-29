import os
import csv
import sys
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

#               clean files in given corpus               #

#        and add all sentences into seperate file         #

# ------------------------------------------------------- #

fn = input("Enter the name of the root folder where all xml files are located: ")
file_name="cleaned_"+fn+".csv"

d=os.path.abspath('..')
listOfFiles = getListOfFiles(d+"/"+fn)
os.chdir(d+"/Output Files")
if os.path.exists(file_name):
	os.remove(file_name)

os.chdir(d+"/Output Files")
f=open(file_name,'w',encoding='utf8',newline ='')
writer = csv.writer(f)
line=[]
no_of_word_tags=0
no_of_files=0
no_of_sentences=0
for file in listOfFiles:
	no_of_files+=1
	tree = ET.parse(file)
	root=tree.getroot()
	for sentence in root.findall('.//s'):
		no_of_sentences+=1
		line.clear()
		for word in sentence.findall('.//w'):
			no_of_word_tags+=1
			line.append(word.text.strip()+'_'+word.attrib['c5'])
				
		if line!=[]:
			writer.writerow(line)


print('No.of files cleaned: ' + str(no_of_files))
print('No.of sentences in dataset: ' + str(no_of_sentences))
print("Number of word_tag's in dataset: " + str(no_of_word_tags))
f.close()
print("All sentences after cleaning are added into \""+file_name+"\" file in the Output Files folder in csv format")

listOfFiles.clear()
line.clear()