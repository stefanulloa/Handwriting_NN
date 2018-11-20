from __future__ import division

import numpy as np
import random



class NeuralNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes # [2,3,1]
        'biases. loop creates a list [] of arrays of size [y,1] for every given layer size, except the input one [1:], which is the 0th position'
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        'zip associates the input layer for x with the second for y [1:] until the previous of the last for x [:-1] with the last one for y'
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]



    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, lr, test_data=None):
        'Stochastic Gradient Descent function, lr is learning rate'
        'test_data can be given to evaluate NN against it in each epoch, but slows learning considerably'

        if test_data: n_test = len(test_data)

        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data) #shuffing occurs at each epoch
            'mini_batches is a list of equally divided training_data from 0 to n using mini_bacth_size for every step'
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

            if test_data:
                print ("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, lr):

        'variables to store the gradients'
        gradient_b = [np.zeros(b.shape) for b in self.biases]
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_gradient_b, delta_gradient_w = self.backprop(x,y)
            'storing computed gradients'
            'gb+dgb every time because this calculates the summatory for all examples x'
            gradient_b = [gb+dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
            gradient_w = [gw+dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

        'updating values with gradient descent'
        'just *gb because it holds the summatory value already done previously'
        self.biases = [b-(lr/len(mini_batch))*gb for b, gb in zip(self.biases, gradient_b)]
        self.weights = [w-(lr/len(mini_batch))*gw for w, gw in zip(self.weights, gradient_w)]

    def backprop(self, x, y):

            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            # feedforward
            activation = x
            activations = [x] # list to store all the activations, layer by layer
            zs = [] # list to store all the z vectors, layer by layer
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            # backward pass
            delta = self.cost_derivative(activations[-1], y) * \
                sigmoid_prime(zs[-1])
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())

            for l in xrange(2, self.num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        'argmax returns index of max activation neuron on ouput layer, in this case it coincides with the position, so argmax can be used for junt this case'
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x==y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
        'derivate of sigmoid function'
        return sigmoid(z)*(1-sigmoid(z))
