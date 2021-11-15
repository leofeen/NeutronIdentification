from prepare_data import Data
from neural_network import OurNeuralNetwork

inputdata = Data('data_w_source.txt', "class_labels.txt")
data = inputdata.get_data()
answers = inputdata.get_answers()

"""data = [data[2], data[18], data[19], data[23], data[24], *data[9:14]]
answers = [1] * 5 + [0] * 5"""

print("Закончена распаковка данных")

for i in range(len(data)):
    for j in range(len(data[i])):
        tmp = max(data[i])
        if data[i][j] == tmp:
            data[i] = data[i][j - 50:j + 450]
            break

print("Закончена обработка данных")
# создание нейронки
network = OurNeuralNetwork(3, 500, 25, 1)


def training():
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
    print("Тренировка окончена")


def practic():
    prac_anss = []
    for i in range(len(data)):
        try:
            tmp = network.feedforward(data[i])
            prac_anss.append(round(tmp))

        except Exception:
            print(f"Возникла непредвиденная ошибка при работе со строкой {i}")

    pers = 0
    lie1 = 0
    lie0 = 0
    for i in range(len(answers)):
        if prac_anss[i] == answers[i]:
            pers += 1
        elif prac_anss[i] == 1 and answers[i] == 0:
            lie1 += 1
        elif prac_anss[i] == 0 and answers[i] == 1:
            lie0 += 1
    print(pers / len(answers) * 100)
    print(f"Ложных нулей: {lie0}, {lie0 / len(answers) * 100} %")
    print(f"Ложных единиц: {lie1}, {lie1 / len(answers) * 100} %")


practic()
