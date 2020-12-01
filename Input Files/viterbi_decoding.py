import numpy as np

def viterbi(y, A, B, words, tags, Pi=None):
    # Cardinality of the state space
	K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
	Pi = Pi if Pi is not None else np.full(K, 1 / K)
	T = len(y)
	T1 = np.empty((K, T), 'd')
	T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
	if y[0] in words:
		T1[:, 0] = Pi * B[:, words.index(y[0])]
	else:
		T1[:,0] = Pi * B[:, -1]
	T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
	for i in range(1, T):
		if y[i] in words:
			T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, words.index(y[i])].T, 1)
		else:
			T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, -1].T, 1)
		T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
	x = np.empty(T, 'B')
	x[-1] = np.argmax(T1[:, T - 1])
	for i in reversed(range(1, T)):
		x[i - 1] = T2[x[i], i]

	return x

'''def Viterbi(words):
	state = []
	for key, word in enumerate(words):
		p = [] 
		for tag in list_of_tags:
			if key == 0:
				transition_p = transition_matrix_df.at['start', tag]
			else:
				transition_p = transition_matrix_df.at[state[-1], tag]

			if words[key] in emission_matrix_df.index:
				emission_p = emission_matrix_df.at[words[key], tag]
			else:
				emission_p=0
			state_probability = emission_p * transition_p   
			p.append(state_probability)
		pmax = max(p)
		state_max = list_of_tags[p.index(pmax)] 
		state.append(state_max)
	return state
'''