import xml.etree.ElementTree as ET

def getWordTagsFromCorpus(listOfFiles):
    count=0
    substring="-"
    tagged_words=[]
    for f in listOfFiles:
        tree = ET.parse(f)
        root=tree.getroot()
        for sentence in root.findall('.//s'):
            for word in sentence.findall('.//w'):
                count=count+1
                if substring in word.attrib['c5']:
                    splittedTags=word.attrib['c5'].split('-')
                    for tag in splittedTags:
                        tagged_words.append(word.text.strip()+'_'+tag)
                else:
                    tagged_words.append(word.text.strip()+'_'+word.attrib['c5'])

    return tagged_words, count;