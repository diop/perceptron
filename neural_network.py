
import numpy as np
np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Dense_Layer(4,5)
layer2 = Dense_Layer(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

some_value = -0.5
weight = 0.7
bias = 0.7

layer_ouputs = [] # Ouput of current layer
for neuron_weights, neuron_bias in zip (weights, biases):
    neuron_output = 0 # Outpur of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_ouput += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(some_value*weight)
print(some_value*bias)

np.random.randn(n_inputs, n_neurons)