import numpy as np

from data_classes import DetectionEvent

def load_data() -> list[DetectionEvent]:
    events = []

    with (open('data_w_source.txt') as data_file, 
        open('class_labels.txt') as labels_file):
        data = data_file.readlines()
        labels = labels_file.readlines()
        for data_str, label_str in zip(data, labels):
            oy = list(map(float, data_str.split()))
            label = label_str == '1'
            events.append(DetectionEvent(oy, label))

    return events