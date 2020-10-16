def countFrequencyOfWordWithtag(my_list):
	freq_of_word_with_tag={}
	for item in my_list:
		splitWords=item.split('_')
		#print(splitWords[0]+' '+splitWords[1])
		if (splitWords[0] in freq_of_word_with_tag):							#if word is already present in dictionary then check if corresponding tag is present or not
			if(splitWords[1] in freq_of_word_with_tag[splitWords[0]]):			#if the tag is already present for the word then increase count
				freq_of_word_with_tag[splitWords[0]][splitWords[1]]+=1
			else:																#if tag is not present then just make the frequency of that tag in that word as 1
				freq_of_word_with_tag[splitWords[0]][splitWords[1]]=1
		else:																	#if word is not present in dictionary, first create a dictionary as value for the key
			freq_of_word_with_tag[splitWords[0]]={}
			freq_of_word_with_tag[splitWords[0]][splitWords[1]]=1
	return freq_of_word_with_tag