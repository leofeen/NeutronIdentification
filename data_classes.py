import numpy as np

from typing import Optional


class DetectionEvent:
    def __init__(self, intesity_points: list, label: Optional[bool] = None):
        self.intensity_points = np.array(intesity_points)
        self.time_axis = np.arange(0, len(intesity_points), 1)
        self.label = label

    def set_label(self, value: bool):
        self.label = value