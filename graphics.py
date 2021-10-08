import matplotlib.pyplot as plt
import numpy as np

from read_files import load_data

events = load_data()

plt.figure(figsize=(20, 16))

j = 1
for event in events:
    if event.label:
        plt.subplot(3, 3, j)
        plt.plot(event.time_axis, event.intensity_points)
        j += 1
        if j == 4:
            break

for event in events:
    if not event.label:
        plt.subplot(3, 3, j)
        plt.plot(event.time_axis, event.intensity_points)
        j += 1
        if j == 7:
            break

for event in events:
    if event.label:
        plt.subplot(3, 3, j)
        plt.plot(event.time_axis, event.intensity_points)
        j += 1
        if j == 10:
            break

plt.show()