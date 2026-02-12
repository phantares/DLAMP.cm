from .registry import SCALER_MAP


class ScalerPipe:
    def __init__(self, stat):
        self.scalers = []

        if stat is not None:
            steps = stat.get("pipeline", [])

            for step in steps:
                scaler_cls = SCALER_MAP.get(step["type"])

                if scaler_cls:
                    params = step.get("params", {})
                    self.scalers.append(scaler_cls(**params))

    def transform(self, data):
        for scaler in self.scalers:
            data = scaler.transform(data)

        return data

    def inverse_transform(self, data):
        for scaler in reversed(self.scalers):
            data = scaler.inverse_transform(data)

        return data
