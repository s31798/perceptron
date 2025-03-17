class Perceptron:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = [0.3,0.3,1]
        self.bias = 0
    def predict(self, x):
        net = 0
        for i in range(len(x)):
            net += self.weights[i] * x[i]
        net -= self.bias
        if net >= 0:
            return 1
        else:
            return 0

    def update_weights(self, x, d):
        y = self.predict(x)
        for i in range(len(x)):
            self.weights[i] = self.weights[i] + self.learning_rate * (d - y) * x[i]
        self.bias = self.bias - self.learning_rate * (d - y)

