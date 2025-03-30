import math
import random
import sys

from matplotlib import pyplot as plt


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
        self.labels = {}

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
        self.d = self.format_data(self.y_train)
        self.vec_length = len(self.x_train[0])
        self.weights = [random.uniform(0, 1) for _ in range(self.vec_length)]

    @staticmethod
    def normalize_vector(vec):
            norm = math.sqrt(sum(x ** 2 for x in vec))
            return [x / norm for x in vec] if norm != 0 else vec


    def format_data(self,y_data):
        d = []
        for i in range(len(y_data)):
            if y_data[i] == y_data[0]:
                d.append(1)
                self.labels[1] = y_data[i]
            else:
                d.append(0)
                self.labels[0] = y_data[i]
        return d

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

    def get_accuracy(self, x_data, y_data):
        correct = 0
        for i in range(len(x_data)):
            if y_data[i] == self.predict(x_data[i]):
                correct += 1
        return correct / len(x_data)

    def learn(self, epochs):
        accuracies = []
        for i in range(epochs):
            iteration_error = 0
            for j in range(len(self.x_train)):
                y = self.update_weights(self.x_train[j], self.d[j])
                iteration_error += (self.d[j] - y) ** 2
            iteration_error = iteration_error / self.n
            accuracy = self.get_accuracy(self.x_train, self.d)
            accuracies.append(accuracy)
            if i % 100 == 0:
                print(f'epoch: {i}, weights: {self.weights}, bias: {self.bias}')
                print(f'accuracy {accuracy}')
        return accuracies


    def eval(self):
        print(f'test accuracy {self.get_accuracy(self.x_test, p.format_data(self.y_test))}')

args = sys.argv
if len(args) != 3 and len(args) != 4:
    raise Exception("Wrong number of arguments")

learning_rate = float(args[1])
epochs = int(args[2])

p = Perceptron(learning_rate)
print(p.weights)
p.open_training_data("perceptron.data")
p.open_test_data("perceptron.test.data")
print(p.x_train)
print(p.d)
p.eval()
accuracies = p.learn(epochs)
p.eval()
if len(args) == 4 and args[3] == "graph":
    plt.plot([i for i in range(epochs)], accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train accuracy after n Epochs")
    plt.show()
else:
    while(True):
            vec = input(f'enter a {p.vec_length}-dimensional vector to label, in the format 1,3,3,...: ').strip().split(',')

            if len(vec) != p.vec_length:
                print("wrong vector length")
                break
            try:
                vec = list(map(float, vec))
                print(p.labels.get(p.predict(vec)))
            except Exception:
                print("please enter a valid vector")