import sys
import numpy as np
import matplotlib as plt
import nnfs

nnfs.init()


def create_data(points, classes):
    x = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for class_num in range(classes):
        ix = range(points * class_num, points * (class_num + 1))
        r = np.linspace(0.0, 1, points)
        t = (
            np.linspace(class_num * 4, (class_num + 1) * 4, points)
            + np.random.randn(points) * 0.2
        )
        x[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_num
    return x, y


# This is my neuron following the tutorial
class Neuron:
    def __init__(self, inputs):
        self.inputs = inputs
        self.weights = [[3.1, 2.1, 8.7], [4.1, 2.1, 4.3], [5.1, 6.3, 7.6]]
        self.biases = [3, 4, 0.3]

    # This function is equivallent to the np.dot function
    def output(self) -> float:
        for self.weights, self.bias in zip(self.weights, self.biases):
            neuron_output = 0
            for n_input, n_weight in zip(self.inputs, self.weights):
                print(n_input, n_weight)
                neuron_output += n_input * n_weight
            neuron_output += self.bias
        return neuron_output

    def get_np_output(self):
        transposed_weights = np.array(self.weights).T

        return np.dot(self.inputs, transposed_weights) + self.biases


# This is a  neuron from the tutorial
class Layer_Dense:
    np.random.seed(0)

    def __init__(self, number_of_inputs, number_of_neurons) -> None:
        self.weights = 0.10 * np.random.randn(number_of_inputs, number_of_neurons)
        self.biases = np.zeros((1, number_of_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLU_Activation:
    np.random.seed(0)

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class SoftMax_Activation:
    def forward(self, inputs):
        batches_exp = np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(inputs - batches_exp)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# def main():
#     x = [[1, 2, 3], [1, 3, 2.3], [2, 3, 2.3]]
#     x_youtube, y_youtube = create_data(100, 3)

#     # neuron = Neuron(inputs=x)
#     # # print(neuron.output())
#     # first_gen_output = neuron.get_np_output()
#     # neuron2 = Neuron(inputs=first_gen_output)
#     # print(first_gen_output)
#     # print(neuron2.get_np_output())

#     layer1 = Layer_Dense(number_of_inputs=2, number_of_neurons=3)
#     # the input must be equal to the output of the previoous layer
#     layer2 = Layer_Dense(number_of_inputs=3, number_of_neurons=3)

#     activation1 = ReLU_Activation()
#     activation2 = SoftMax_Activation()
#     print(activation2)

#     loss_function = Loss_CategoricalCrossEntropy()

#     layer1.forward(x_youtube)
#     activation1.forward(layer1.output)
#     layer2.forward(activation1.output)
#     activation2.forward(layer2.output)

#     print(activation2.output[:1], [0, 1, 2])
#     loss = loss_function.calculate(activation2.output, y_youtube)
#     print("Loss:", loss)


# print(layer2.output)


if __name__ == "__main__":
    main()
