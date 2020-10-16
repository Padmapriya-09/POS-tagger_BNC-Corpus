import xml.etree.ElementTree as ET

def trainCorpus(listOfFiles):
	tagged_words=[]
	for f in listOfFiles:	
		train_file = open(f)
		tree = ET.parse(train_file)
		root=tree.getroot()
		for word in root.findall('.//w'):										#finds only elements with a tag which are direct children of the current element
			tagged_words.append(word.text.strip()+'_'+word.attrib['pos'])

	return tagged_words