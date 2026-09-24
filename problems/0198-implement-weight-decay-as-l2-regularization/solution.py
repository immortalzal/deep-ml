def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	for i in range(len(parameters)):
		decay = 1
		if(apply_to_all[i]):
			decay = 1 - weight_decay * lr
		for j in range(len(parameters[i])):
			parameters[i][j] = decay * parameters[i][j] - lr * gradients[i][j]
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	return parameters
	# Your code here
	pass