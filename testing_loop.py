import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
from main import *
nnfs.init()

X,y = spiral_data(100,3)


dense1 = Layer_Dense(2,3)
activation1 = ReLU_Activation()
dense2 = Layer_Dense(3,3)
activation2 = SoftMax_Activation()

loss_func = Loss_CategoricalCrossEntropy()

lowest_loss: int = 99999


best_dense1_weights = dense1.weights.copy()
best_dense1_bias = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_bias = dense2.biases.copy()

for iteraction in range(100_000):
	
	# Totally random sets every iter
	# dense1.weights = 0.05 *np.random.randn(2,3)
	# dense1.biases = 0.05 *np.random.randn(1,3)
	# dense2.weights = 0.05 *np.random.randn(3,3)
	# dense2.biases = 0.05 *np.random.randn(1,3)

	# Some tweaks over the best know set
	dense1.weights += 0.05 *np.random.randn(2,3)
	dense1.biases += 0.05 *np.random.randn(1,3)
	dense2.weights += 0.05 *np.random.randn(3,3)
	dense2.biases += 0.05 *np.random.randn(1,3)

	dense1.forward(X)
	activation1.forward(dense1.output)
	dense2.forward(activation1.output)
	activation2.forward(dense2.output)

	loss = loss_func.calculate(activation2.output, y)

	predictions = np.argmax(activation2.output, axis=1)
	accuracy = np.mean(predictions==y)

	if loss < lowest_loss:
		print(f"New set of weights and biases iter: {iteraction}, loss: {loss}, acc: {accuracy}")
		best_dense1_weights = dense1.weights.copy()
		best_dense1_bias = dense1.biases.copy()
		best_dense2_weights = dense2.weights.copy()
		best_dense2_bias = dense2.biases.copy()
		lowest_loss = loss
	else:

		dense1.weights  = best_dense1_weights.copy()
		dense1.biases = best_dense1_bias.copy()
		dense2.weights = best_dense2_weights.copy()
		dense2.biases = best_dense2_bias.copy()
