import matplotlib.pyplot as plt
import numpy as np

oys = [] # значения на oy

with open('data_w_source.txt') as data_file:
    data = data_file.readlines()
    for i in range(0, len(data)):
        oys.append(list(map(float, data[i].split())))


ox = np.arange(0, 3735, 1)

plt.figure(figsize=(20, 16))
plt.subplot(211)
plt.plot(ox, oys[0])
plt.subplot(212)
plt.plot(ox, oys[2])
plt.show()