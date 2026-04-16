class ZScoreScaler:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def transform(self, data):
        if self.mean is not None and self.std is not None:
            if self.std != 0:
                data = (data - self.mean) / self.std

        return data

    def inverse_transform(self, data):
        if self.mean is not None and self.std is not None:
            device = data.device

            if self.std != 0:
                data = data * self.std.to(device) + self.mean.to(device)

        return data
