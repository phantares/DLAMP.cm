from pathlib import Path
from dotenv import dotenv_values
from collections import defaultdict
import json

from . import ScalerPipe, IdentityScaler


def get_scaler_map(stats_file):
    env = dotenv_values(".env")

    with open(Path(env.get("STATS_DIR"), stats_file), "r") as f:
        stats = json.load(f)

    scaler_map = defaultdict(IdentityScaler)
    for var, s in stats.items():
        scaler_map[var] = ScalerPipe(s)

    return scaler_map
