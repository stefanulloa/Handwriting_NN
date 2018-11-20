import numpy as np

class NeuralNetwork(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes # [2,3,1]
        'biases. loop creates a list [] of arrays of size [y,1] for every given layer size, except the input one [1:], which is the 0th position'
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        'zip associates the input layer for x with the second for y [1:] until the previous of the last for x [:-1] with the last one for y'
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

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
        mini_batches = [training_data[k:+mini_batch_size] for k in xrange[0,n,mini_batch_size]]
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
        gradient_b = [gb+dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
        gradient_w = [gw+dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

    'updating values with gradient descent'
    self.biases = [b-(ln/len(mini_batch)*gb for b, gb in zip(self.biases, gradient_b))]
    self.weights = [w-(ln/len(mini_batch)*gw for w, gw in zip(self.weights, gradient_w))]
