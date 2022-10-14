# this program impements a neural network from scratch only using numpy 

import numpy as np


# NEURAL NETWORK CLASS
class NeuralNetwork:
    def __init__(self, input_size, learning_rate=0.1):
        self.input_size = input_size
        self.weights = []
        self.bias = []
        self.learning_rate = learning_rate
        self.activations = []
        self.nodes = []

    # add a layer to the network
    def add_layer(self, num_nodes, activation_function):
        # if this is the first layer then set the weights to the input size
        if len(self.weights) == 0:
            self.weights.append(np.random.randn(self.input_size, num_nodes))

        else:
            self.weights.append(np.random.randn(self.nodes[-1].shape[1], num_nodes))

        # add the bias
        self.bias.append(np.random.randn(1, num_nodes))

        # add the activation function
        self.activations.append(activation_function)

        # add the nodes 
        self.nodes.append(np.zeros((1, num_nodes)))


    # feed forward throught the network to make a predicition
    def feed_forward(self, inputs, training=True):
        # for each layer in the network
        for i in range(len(self.weights)):
            # calculate the nodes
            inputs = np.dot(inputs, self.weights[i]) + self.bias[i]

            inputs = self.activations[i](inputs)

            if training:
                self.nodes[i] = inputs
        
        return inputs


    # back propagate the error to train the network
    def back_propagate(self, actual):
        # calculate the error
        error = actual - self.nodes[-1]

        # calculate the gradients
        gradients = [error * self.activations[-1](self.nodes[-1], derivative=True)]

        # for each layer in the network
        for i in range(len(self.weights) - 2, -1, -1):
            # calculate the error
            error = np.dot(gradients[-1], self.weights[i + 1].T)

            # calculate the gradients
            gradients.append(error * self.activations[i](self.nodes[i], derivative=True))
    
        # reverse the gradients so they are in the correct order
        gradients.reverse()

        # update the weights
        self.update_weights(gradients)


    def update_weights(self, gradients):
        # for each layer in the network
        for i in range(len(self.weights)):
            # update the weights
            self.weights[i] += self.learning_rate * np.dot(self.nodes[i], gradients[i].T)

            # update the bias
            self.bias[i] += self.learning_rate * gradients[i]


    def train(self, inputs, expected_outputs):
        # feed the inputs into the network
        out = self.feed_forward(inputs)

        # back propagate the error to train the network
        self.back_propagate(expected_outputs)


    def test(self, inputs, expected_outputs):
        # feed the inputs into the network
        accuracy = 0

        for i in range(len(inputs)):
            out = self.feed_forward(inputs[i], training=False)

            if np.argmax(out) == np.argmax(expected_outputs[i]):
                accuracy += 1

        # return the accuracy
        return accuracy / len(inputs) * 100


    def save(self, filename):
        # save the weights to a file
        np.save(filename, self.weights)

    
    def load(self, filename):
        # load the weights from a file
        self.weights = np.load(filename, allow_pickle=True)


# ACTIVATION FUNCTIONS

# calculate the sigmoid activation function for a numpy array
def sigmoid(x, derivative=False):
    # calculate the sigmoid function
    if derivative:
        # calculate the derivative of the sigmoid function
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


# calulate the leaky relu activation function for a numpy array
def leakyReLU(x, derivative=False, alpha=0.01):
    # calculate the leaky relu function
    if derivative:
        # calculate the derivative of the leaky relu function
        x[x <= 0] = alpha
        x[x > 0] = 1
        return x
    else:
        x[x <= 0] *= alpha
        return x


def softmax(x, derivative=False):
    # calculate the softmax function
    if derivative:
        # calculate the derivative of the softmax function
        return x * (1 - x)
    else:
        exponent = np.exp(x - np.max(x))
        return exponent / exponent.sum()


# DRIVER CODE
def main():
    # create a new neural network
    nn = NeuralNetwork(2, 0.8)

    # add 2 hidden layers with a leaky relu activation function
    nn.add_layer(50, sigmoid)

    nn.add_layer(50, sigmoid)

    # add the output layer with a softmax function
    nn.add_layer(2, softmax)
    

    # create some training data to test if the network can learn the XOR function
    train_x = np.array([[1, 1], [1, 1], [0, 0], [0, 0], [1, 0], [0, 1], [1, 0], [0, 1]])
    train_y = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])

    # training loop
    EPOCHS = 1000
    for i in range(EPOCHS):
        for j in range(len(train_x)):
            nn.train(train_x[j], train_y[j])

    # create some training data
    test_x = np.array([[1, 1], [1, 1], [0, 0], [0, 0], [1, 0], [0, 1], [1, 0], [0, 1]])
    test_y = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])

    accuracy = nn.test(test_x, test_y)
    print(f'Accuracy: {accuracy}%')

if __name__ == "__main__":
    main()

