import numpy as np
import math
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	mse_values = [0] * epochs
	for k in range(epochs):

		probabilities = [0] * len(features)
		z_val = [0] * len(features)

		for i in range(len(features)):
			sum = 0
			for j in range(len(initial_weights)):
				sum += features[i][j] * initial_weights[j]
			probabilities[i] = sum + initial_bias

		for i in range(len(probabilities)):
			z_val[i] = probabilities[i]

		for i in range(len(probabilities)):
			probabilities[i] = 1 / (1 + math.exp(-probabilities[i]))

		mse = 0

		for i in range(len(probabilities)):
			mse += (probabilities[i] - labels[i]) ** 2

		mse /= len(probabilities)

		mse_values[k] = mse

		for j in range(len(initial_weights)):
			deriv = 0
			for i in range(len(probabilities)):
				deriv += (probabilities[i] - labels[i]) * math.exp(-z_val[i]) / ((1 + math.exp(-z_val[i])) ** 2) * features[i][j]
			deriv *= 2
			deriv /= len(probabilities)
			initial_weights[j] -= learning_rate * deriv

		deriv = 0

		for i in range(len(probabilities)):
			deriv += (probabilities[i] - labels[i]) * math.exp(-z_val[i]) / ((1 + math.exp(-z_val[i])) ** 2)
		deriv *= 2
		deriv /= len(probabilities)

		initial_bias -= learning_rate * deriv

	updated_weights = initial_weights

	updated_bias = initial_bias	

	return updated_weights, updated_bias, mse_values