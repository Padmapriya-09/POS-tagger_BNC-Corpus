def findCompleteLengthOfNestedDictionary(dict):
	length=0
	for key,value in dict.items():
		length += len(dict[key])
	return length