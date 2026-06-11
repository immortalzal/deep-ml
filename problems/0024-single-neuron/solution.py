import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	probabilities = [0] * len(features)
	for i in range(len(features)):
		sum = 0
		for j in range(len(weights)):
			sum += features[i][j] * weights[j]
		probabilities[i] = sum + bias
	for i in range(len(probabilities)):
		probabilities[i] = 1 / (1 + math.exp(-probabilities[i]))
	mse = 0
	for i in range(len(probabilities)):
		mse += (probabilities[i] - labels[i]) ** 2
	mse /= len(probabilities)
	return probabilities, mse