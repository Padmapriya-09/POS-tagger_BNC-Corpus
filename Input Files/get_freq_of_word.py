def countFrequencyOfWord(my_list):  
	freq_of_word = {}
	for item in my_list:
		splitWords=item.split('_')
		if (splitWords[0] in freq_of_word): 
			freq_of_word[splitWords[0]] += 1
		else:
			freq_of_word[splitWords[0]] = 1
	return freq_of_word