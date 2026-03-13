import numpy as np
import torch


class LogScaler:
    def __init__(self, ref=None):
        self.ref = ref

    def transform(self, data):
        if self.ref is None:
            return data

        if torch.is_tensor(data):
            return torch.log10(data / self.ref + 1)
        else:
            return np.log10(data / self.ref + 1)

    def inverse_transform(self, data):
        if self.ref is None:
            return data

        if torch.is_tensor(data):
            return (torch.pow(10, data) - 1) * self.ref
        else:
            return (np.power(10, data) - 1) * self.ref

