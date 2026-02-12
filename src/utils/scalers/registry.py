from . import ZScoreScaler, MinMaxScaler, LogScaler

SCALER_MAP = {
    "zscore": ZScoreScaler,
    "minmax": MinMaxScaler,
    "log": LogScaler,
}
