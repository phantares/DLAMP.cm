from pathlib import Path
from dotenv import dotenv_values
import json

from . import ScalerPipe


def get_scaler_map(stats_file):
    env = dotenv_values(".env")

    with open(Path(env.get("STATS_DIR"), stats_file), "r") as f:
        stats = json.load(f)

    scaler_map = {var: ScalerPipe(s) for var, s in stats.items()}

    return scaler_map
