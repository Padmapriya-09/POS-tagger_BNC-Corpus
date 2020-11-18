import xml.etree.ElementTree as ET

def getWordTagsFromCorpus(listOfFiles):
	count=0
	substring="-"
	tagged_words=[]
	for f in listOfFiles:	
		train_file = open(f)
		tree = ET.parse(train_file)
		root=tree.getroot()
		for word in root.findall('.//w'):										#finds only elements with a tag which are direct children of the current element
			count=count+1
			if substring in word.attrib['c5']:
				splittedTags=word.attrib['c5'].split('-')
				for tag in splittedTags:
					tagged_words.append(word.text.strip()+'_'+tag)
			else:
				tagged_words.append(word.text.strip()+'_'+word.attrib['c5'])

	return tagged_words, count;