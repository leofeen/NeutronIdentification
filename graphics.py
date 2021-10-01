import matplotlib.pyplot as plt
import numpy as np

oys = [] # значения на oy
labels = []

with open('data_w_source.txt') as data_file:
    data = data_file.readlines()
    for i in range(0, len(data)):
        oys.append(list(map(float, data[i].split())))

with open('class_labels.txt') as data_file:
    data = data_file.readlines()
    for i in range(0, len(data)):
        labels.append(int(data[i]))


ox = np.arange(0, 3735, 1)

plt.figure(figsize=(20, 16))

j = 1
for data, label in zip(oys, labels):
    if label == 1:
        plt.subplot(3, 3, j)
        plt.plot(ox, data)
        j += 1
        if j == 4:
            break

for data, label in zip(oys, labels):
    if label == 0:
        plt.subplot(3, 3, j)
        plt.plot(ox, data)
        j += 1
        if j == 7:
            break

for data, label in zip(oys, labels):
    if label == 1:
        plt.subplot(3, 3, j)
        plt.plot(ox, data)
        j += 1
        if j == 10:
            break

plt.show()