def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	if len(a[0]) != len(b):
		return -1
	arr = [0] * len(a)
	for i in range(len(a)):
		sum = 0
		for j in range(len(b)):
			sum += a[i][j] * b[j]
		arr[i] = sum
	return arr
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	pass