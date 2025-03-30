import math
import random
class Perceptron:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = random.uniform(0, 1)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.vec_length = None
        self.d = None
        self.n = 0

    @staticmethod
    def load_data_from_file(file_name):
        x = []
        y = []
        n = 0
        try:
            file = open("data/" + file_name)
            line = file.readline()
            while line:
                vec = []
                line = line.strip().split(",")
                for i in range(len(line) - 1):
                    vec.append(float(line[i]))
                x.append(vec)
                y.append(line[-1])
                n += 1
                line = file.readline()
            return x, y, n
        except FileNotFoundError:
            raise Exception(f'file {file_name} not found')

    def open_training_data(self, file_name):
        self.x_train,self.y_train, self.n = Perceptron.load_data_from_file(file_name)
        self.d = []
        for i in range(len(self.y_train)):
            if self.y_train[i] == self.y_train[0]:
                self.d.append(1)
            else:
                self.d.append(0)

        self.vec_length = len(self.x_train[0])
        self.weights = [random.uniform(0, 1) for _ in range(self.vec_length)]

    @staticmethod
    def normalize_vector(vec):
            norm = math.sqrt(sum(x ** 2 for x in vec))
            return [x / norm for x in vec] if norm != 0 else vec


    def format_data(self):
        self.d = []
        for i in range(len(self.y_train)):
            if self.y_train[i] == self.y_train[0]:
                self.d.append(1)
            else:
                self.d.append(0)

    def open_test_data(self, file_name):
        self.x_test, self.y_test, _= Perceptron.load_data_from_file(file_name)

    def predict(self, x_vec):
        Perceptron.normalize_vector(x_vec)
        net = 0
        for i in range(len(x_vec)):
            net += self.weights[i] * x_vec[i]
        net -= self.bias
        if net >= 0:
            return 1
        else:
            return 0

    def update_weights(self, x_vec, d):
        y = self.predict(x_vec)
        for i in range(len(x_vec)):
            self.weights[i] = self.weights[i] + self.learning_rate * (d - y) * x_vec[i]
        self.bias = self.bias - self.learning_rate * (d - y)
        return y

    def get_training_accuracy(self):
        correct = 0
        for i in range(self.n):
            if self.d[i] == self.predict(self.x_train[i]):
                correct += 1
        return correct / self.n



    def learn(self, epochs):
        for i in range(epochs):
            iteration_error = 0
            for j in range(len(self.x_train)):
                y = self.update_weights(self.x_train[j], self.d[j])
                iteration_error += (self.d[j] - y) ** 2
            iteration_error = iteration_error / self.n
            ##print(self.weights)
            if i % 100 == 0:
                print(f'epoch: {i}, error: {iteration_error}')
                print(f'accuracy {self.get_training_accuracy()}')




p = Perceptron(learning_rate=0.01)
print(p.weights)
p.open_training_data("perceptron.data")
p.open_test_data("perceptron.test.data")
print(p.x_train)
print(p.d)
p.learn(epochs=1000)