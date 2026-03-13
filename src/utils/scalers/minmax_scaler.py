class MinMaxScaler:
    def __init__(self, min=None, max=None):
        self.min = min

        if min is not None and max is not None and max > min:
            self.range = max - min
        else: 
            self.range= None

    def transform(self, data):
        if self.range is None:
            return data

        return (data - self.min) / self.range * 2.0 - 1.0

    def inverse_transform(self, data):
        if self.range is None:
            return data

        return (data + 1.0) / 2.0 * self.range + self.min
