import numpy as np

from typing import Optional


class DetectionEvent:
    def __init__(self, intesity_points: np.ndarray, label: Optional[bool] = None):
        self.intensity_points = intesity_points
        self.time_axis = np.arange(0, intesity_points.size, 1)
        self.label = label

    def set_label(self, value: bool):
        self.label = value