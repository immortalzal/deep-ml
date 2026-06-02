import numpy as np

def hamming_distance_kanerva(state: list, prototypes: list, threshold: int) -> tuple:
	v = []
	ans = [0] * len(prototypes)
	for i in range(len(prototypes)):
		sum = 0
		for j in range(len(state)):
			if state[j] != prototypes[i][j]:
				sum += 1
		ans[i] = sum
	for i in range(len(prototypes)):
		if ans[i] <= threshold:
			v.append(i)
	return(ans, v)
	pass