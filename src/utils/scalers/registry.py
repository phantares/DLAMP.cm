from .log_scaler import LogScaler
from .minmax_scaler import MinMaxScaler
from .z_score_scaler import ZScoreScaler

SCALER_MAP = {
    "zscore": ZScoreScaler,
    "minmax": MinMaxScaler,
    "log": LogScaler,
}
