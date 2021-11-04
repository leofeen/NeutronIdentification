from prepare_data import Data
from neural_network import OurNeuralNetwork

"""inputdata = Data('data_w_source.txt', "class_labels.txt")
data = inputdata.get_data()
answers = inputdata.get_answers()"""

data = [[-20, -10], [5, 6], [1, 1], [-1, -1]]
answers = [0, 1, 1, 0]

# создание нейронки
network = OurNeuralNetwork(3, 2, 2, 1)

# процесс тренировки (при дате без ответов закомментить)
network.train(data, answers)

with open("weights.txt", "w") as f:
    for i in network.weights:
        for j in i:
            f.write(" ".join(map(str, j)) + "\n")
        f.write("__\n")

with open("biases.txt", "w") as f:
    for i in network.layers:
        s = []
        for j in i.neurons:
            s.append(j.bias)
        f.write(" ".join(map(str, s)) + "\n")

for i in data:
    print(network.feedforward(i))
