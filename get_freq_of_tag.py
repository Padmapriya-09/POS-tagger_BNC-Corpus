def countFrequencyOfTag(my_list):
	freq_of_tag={}
	for item in my_list:
		splitWords=item.split('_')
		if(splitWords[1] in freq_of_tag):
			freq_of_tag[splitWords[1]] += 1
		else:
			freq_of_tag[splitWords[1]] = 1
	return freq_of_tag