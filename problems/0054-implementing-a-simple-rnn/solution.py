import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	h_t = initial_hidden_state
	for i in range(len(input_sequence)):
		h_t = np.tanh(np.matmul(Wx, input_sequence[i]) + np.matmul(Wh, h_t) + b)
	# Your code here
	return h_t