from prepare_data import Data
from neural_network import OurNeuralNetwork

inputdata = Data('data_w_source.txt', "class_labels.txt")
data = inputdata.get_data()
answers = inputdata.get_answers()

# создание нейронки
network = OurNeuralNetwork(4, 3735, 100, 10, 1)
with open("weights.txt", "w") as f:
    for i in network.weights:
        for j in i:
            f.write(" ".join(map(str, j)) + "\n")

network.train(data, answers)

with open("weights2.txt", "w") as f:
    for i in network.weights:
        for j in i:
            f.write(" ".join(map(str, j)) + "\n")

for i in data:
    print(network.feedforward(i))
