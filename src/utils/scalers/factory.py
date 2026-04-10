from pathlib import Path
from dotenv import dotenv_values
import re
from collections import defaultdict
import json
import torch

from . import SCALER_MAP, LogScaler, MinMaxScaler, ZScoreScaler


def get_scaler_map(stats_file, **level_variables):
    env = dotenv_values(".env")

    with open(Path(env.get("STATS_DIR"), stats_file), "r") as f:
        stats = json.load(f)

    scaler_map = defaultdict(IdentityScaler)
    for var, s in stats.items():
        scaler_map[var] = ScalerPipe(s)
    keys = scaler_map.keys()

    for variable, levels in level_variables.items():
        var_exp = [k for k in keys if k.startswith(variable)][0]

        match = re.match(r"^([a-zA-Z_]+)(\d+)$", var_exp)
        if match:
            pipelines = [
                scaler_map.get(f"{variable}{int(z)}", ScalerPipe(None)) for z in levels
            ]
            scaler_map[variable] = StackedScalerPipe(pipelines)

    return scaler_map


class IdentityScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


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


class StackedScalerPipe:
    def __init__(self, pipelines):
        self.levels = len(pipelines)

        non_empty = [(i, p) for i, p in enumerate(pipelines) if len(p.scalers) > 0]
        self.scaled_indices = torch.tensor([i for i, _ in non_empty])

        all_zero = [(i, p) for i, p in enumerate(pipelines) if len(p.scalers) == 0]
        self.all_zero_indices = torch.tensor([i for i, _ in all_zero])
        self.last_mm = False

        scalers = []
        n_steps = len(non_empty[0][1].scalers)
        for step_idx in range(n_steps):
            step_scalers = [p.scalers[step_idx] for _, p in non_empty]
            scaler_type = type(step_scalers[0])

            if scaler_type == LogScaler:
                ref = torch.tensor([s.ref for s in step_scalers])[:, None, None]
                scalers.append(LogScaler(ref=ref))

            elif scaler_type == MinMaxScaler:
                mn = torch.tensor([s.min for s in step_scalers])[:, None, None]
                mx = torch.tensor([s.min + s.range for s in step_scalers])[
                    :, None, None
                ]
                scalers.append(MinMaxScaler(min=mn, max=mx))

                if step_idx == n_steps - 1:
                    self.last_mm = True

            elif scaler_type == ZScoreScaler:
                mean = torch.tensor([s.mean for s in step_scalers])[:, None, None]
                std = torch.tensor([s.std for s in step_scalers])[:, None, None]
                scalers.append(ZScoreScaler(mean=mean, std=std))

        self.pipeline = ScalerPipe(None)
        self.pipeline.scalers = scalers

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        idx = self.scaled_indices.to(data.device)
        scaled_data = self.pipeline.transform(data[..., idx, :, :])
        data[..., idx, :, :] = scaled_data

        if self.last_mm and len(self.all_zero_indices) > 0:
            idx = self.all_zero_indices.to(data.device)
            all_zero_data = data[..., idx, :, :] - 1.0
            data[..., idx, :, :] = all_zero_data

        return data

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        idx = self.scaled_indices.to(data.device)
        invt_data = self.pipeline.inverse_transform(data[..., idx, :, :])
        data[..., idx, :, :] = invt_data

        if self.last_mm and len(self.all_zero_indices) > 0:
            idx = self.all_zero_indices.to(data.device)
            all_zero_data = data[..., idx, :, :] + 1.0
            data[..., idx, :, :] = all_zero_data

        return data
