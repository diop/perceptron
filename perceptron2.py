import numpy as np

class Perceptron:
    def __init__(self, n_features=3, weights = None, bias=0, lr=0.1, epochs=100):
        if weights==None:
            self.weights = np.zeros(n_features)
        else :
            self.weights=weights
        self.epochs = epochs
        self.bias = bias
        self.lr = lr

    def weighted_sum(self, x):
        ws = np.dot(x, self.weights) + self.bias
        return ws

    def activation(self, weighted_sum):
        return (weighted_sum >= 0).astype(int)

    def predict(self, x):
        return self.activation(self.weighted_sum(x))

    def compute_error(self, X, y):
        error = y - self.predict(X)
        return error
    
    def update_weights(self, X, error):
        for x, e in zip(X, error):
            self.weights += self.lr * e * x
            self.bias += self.lr * e
    
    def fit(self, X, y):
        e = 0
        for _ in range(self.epochs):
            errors = self.compute_error(X, y)
            self.update_weights(X, errors)
            print(f'epoch {_}: error {np.mean(errors)}')

percep = Perceptron()
print(percep)