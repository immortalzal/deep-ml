import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)
