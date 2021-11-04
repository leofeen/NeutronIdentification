import numpy as np
from random import random


def sigmoid(x):
    # Функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, lay, pos):
        self.amount = 0
        self.value = 0
        self.lay_number = lay
        self.number = pos
        self.bias = 0
        self.weights = []

    def count_value(self, weights, previous_layer):
        self.weights = weights
        tmp = 0
        for i in range(len(previous_layer.neurons)):
            tmp += weights[i] * previous_layer.neurons[i].value
        self.amount = tmp + self.bias
        self.value = sigmoid(self.amount)
        return self.value


class FirstNeuron(Neuron):
    def __init__(self, lay, pos, value):
        self.bias = 0
        self.lay_number = 0
        self.number = pos
        self.amount = value + self.bias
        self.value = sigmoid(self.amount)
        self.weights = []

    def count_value(self, weights, previous_layer):
        return self.value


class Layer:
    def __init__(self, number_of_layer, number_of_neurons, previous_layer):
        self.neurons = [Neuron(number_of_layer, q) for q in range(number_of_neurons)]
        self.previous_layer = previous_layer

    def count_values(self, weights):
        for q in range(len(self.neurons)):
            self.neurons[q].count_value(weights[q], self.previous_layer)

    def __len__(self):
        return len(self.neurons)


class FirstLayer(Layer):
    def __init__(self, number_of_layer, number_of_neurons, data):
        self.neurons = [FirstNeuron(0, q, data[q]) for q in range(number_of_neurons)]

    def count_values(self, weights):
        pass


class OurNeuralNetwork:

    def __init__(self, num_of_layers, *number_of_neurons_in_layers):
        """инициализация нейронки, количество слоёв и количество нейронов в каждом слое"""
        self.layers = [FirstLayer(0, number_of_neurons_in_layers[0], [0] * number_of_neurons_in_layers[0])]
        for i in range(1, num_of_layers):
            self.layers.append(Layer(i, number_of_neurons_in_layers[i], self.layers[-1]))

        # создание случайных весов, которые потом поменяются
        self.weights = [[]]
        for i in range(1, len(self.layers)):
            self.weights.append(
                [[random() for _ in range(number_of_neurons_in_layers[i - 1])] for _ in
                 range(number_of_neurons_in_layers[i])])

    # просто функция, закидываешь данные, возвращается ответ
    def feedforward(self, data):
        if len(self.layers[0]) != len(data):
            raise Exception("Количество данных должно совпадать с количеством нейронов первого слоя!")
        self.layers[0] = FirstLayer(0, len(data), data)
        for q in range(len(self.layers)):
            self.layers[q].count_values(self.weights[q])
        return self.layers[-1].neurons[0].value

    # функция пока не переделана!!!
    def train(self, data, right_answers):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
            Elements in all_y_trues correspond to those in data.
        """
        learn_rate = 0.5
        epochs = 50  # количество циклов во всём наборе данных

        for epoch in range(epochs):

            for x, y_true in zip(data, right_answers):

                """"# --- Выполняем обратную связь (нам понадобятся эти значения в дальнейшем)
                neurons = [[] * len(self.layers)]
                neurons[0] = [(self.layers[0].neurons[i].value, self.layers[0].neurons[i].amount) for i in
                              range(len(self.layers[0]))]
                for i in range(1, len(self.layers)):
                    for j in range(len(self.layers[i])):
                        neurons[i][j] = (self.layers[i].neurons[j].count_value(self.weights[i][j], self.layers[i - 1]),
                                         self.layers[i].neurons[j].amount)

                y_pred = neurons[-1][0][0]"""

                y_pred = self.feedforward(x)

                dL_dypred = -2 * (y_true - y_pred)

                for layer in self.layers[1:]:
                    dw = learn_rate * dL_dypred
                    for neuron in layer.neurons:
                        num_of_lay = neuron.lay_number
                        dw *= deriv_sigmoid(neuron.value)
                        for num_weight in range(len(neuron.weights)):
                            neuron.weights[num_weight] -= dw * self.layers[num_of_lay - 1].neurons[num_weight].amount

                """"# --- Подсчет частных производных
                # --- Наименование: d_L_d_w1 представляет "производная L по w1"
                dL_dypred = -2 * (y_true - y_pred)

                dypred_dw = []
                dypred_db = []"""

                """# --- Выполняем обратную связь (нам понадобятся эти значения в дальнейшем)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Подсчет частных производных
                # --- Наименование: d_L_d_w1 представляет "частично L / частично w1"
                d_L_d_ypred = -2 * (y_true - y_pred)
                
                # Нейрон o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Обновляем вес и смещения
                # Нейрон h1
                self.w1 -= learn_rate * dL_dypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * dL_dypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * dL_dypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * dL_dypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * dL_dypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * dL_dypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w5 -= learn_rate * dL_dypred * d_ypred_d_w5
                self.w6 -= learn_rate * dL_dypred * d_ypred_d_w6
                self.b3 -= learn_rate * dL_dypred * d_ypred_d_b3

            # --- Подсчитываем общую потерю в конце каждой фазы
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)"""


""" # Определение набора данных
data = np.array([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6]

])

all_y_trues = np.array([
    0,  # Alice
    1,  # Bob
    1,  # Charlie
    0,  # Diana
])

# Тренируем нашу нейронную сеть!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
O = []
for i in range(len(data)):
    O.append(network.feedforward(data[i]))
output = np.array(O).T
print(output)"""

"""
network = OurNeuralNetwork(3, 4, 5, 1)
print(network.feedforward([1, 2, 3, 4]))
"""  # пример использования нейронки: в нашем случае количество нейронов последнего слоя должно быть равно 1
