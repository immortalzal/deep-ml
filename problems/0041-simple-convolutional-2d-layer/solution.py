import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	final_input = np.pad(input_matrix, pad_width = padding, mode = 'constant', constant_values = 0)
	h = (input_height + 2 * padding - kernel_height + 1) // stride 
	w = (input_width + 2 * padding - kernel_width + 1) // stride 
	output_matrix = np.zeros((h, w), dtype = float)
	for i in range(0, final_input.shape[0] - kernel_height + 1, stride):
		for j in range(0, final_input.shape[1] - kernel_width + 1, stride):
			for k in range(kernel_height):
				for l in range(kernel_width):
					output_matrix[i // stride][j // stride] += kernel[k][l] * final_input[i + k][j + l]
	# Your code here   
	return output_matrix
