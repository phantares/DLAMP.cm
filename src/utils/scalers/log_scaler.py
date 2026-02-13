import numpy as np


class LogScaler:
    def __init__(self, ref=None):
        self.ref = ref

    def transform(self, data):
        if self.ref is not None:
            data = np.log10(data / self.ref + 1)

        return data

    def inverse_transform(self, data):
        if self.ref is not None:
            data = (np.power(10, data) - 1) * self.ref

        return data
