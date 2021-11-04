from prepare_data import Data
from neural_network import OurNeuralNetwork

"""inputdata = Data('data_w_source.txt', "class_labels.txt")
data = inputdata.get_data()
answers = inputdata.get_answers()"""

data = [[-2, -1], [25, 6], [17, 4], [-15, -6]]
answers = [0, 1, 1, 0]

# создание нейронки
network = OurNeuralNetwork(3, 2, 2, 1)

# при самом первом запуске нейронки создать файл weights.txt и запустить эту строку
# network.create_new_weights()

network.train(data, answers)

with open("weights.txt", "w") as f:
    for i in network.weights:
        for j in i:
            f.write(" ".join(map(str, j)) + "\n")
        f.write("__\n")

for i in data:
    print(network.feedforward(i))
