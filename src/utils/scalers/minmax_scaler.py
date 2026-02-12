class MinMaxScaler:
    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max

    def transform(self, data):
        if self.min is not None and self.max is not None:
            if self.max > self.min:
                data = (data - self.min) / (self.max - self.min) * 2 - 1.0

        return data

    def inverse_transform(self, data):
        if self.min is not None and self.max is not None:
            if self.max > self.min:
                data = (data + 1.0) / 2.0 * (self.max - self.min) + self.min

        return data
