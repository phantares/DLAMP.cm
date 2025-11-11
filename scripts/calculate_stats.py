from pathlib import Path
from dotenv import dotenv_values
import argparse
import h5py as h5
import numpy as np
import json
import torch
import torch.nn.functional as F


variables_constant = ["terrain", "latitude", "longitude"]
variables_single = ["sst", "psfc", "t2m", "q2m", "u10", "v10", "olr", "sw"]
variables_upper = ["z", "u", "v", "w", "t", "qv", "qc", "qr", "qi", "qs", "qg"]


def main(resolution, shape, input_dir=None, output_dir=None):
    env = dotenv_values(".env")

    dir = Path(input_dir or env.get("INPUT_DIR"))
    files = sorted(dir.glob("*.h5"))
    results = {}

    for variable in variables_constant:
        with h5.File(files[0], "r") as f:
            var = _to_4d(np.array(f[variable][:]))
            var = F.interpolate(var, size=shape, mode="bilinear")

            mean = torch.mean(var)
            std = torch.std(var)
            results[variable] = {"mean": float(mean), "std": float(std)}

    for variable in variables_single:
        sum_all = 0.0
        sumsq_all = 0.0
        count_all = 0.0

        for file in files:
            with h5.File(file, "r") as f:
                var = _to_4d(np.array(f[variable[:]]))
                var = F.interpolate(var, size=shape, mode="bilinear")

                if variable == "sst":
                    mask = (var > 0).float()
                    den = F.interpolate(mask, size=shape, mode="bilinear").clamp_min(
                        1e-6
                    )
                    var = var / den
                    var[var <= 0.0] = float("nan")

                mask = torch.isfinite(var)
                var = var[mask]

                sum_all += torch.sum(var)
                sumsq_all += torch.sum(var**2)
                count_all += var.numel()

        mean = sum_all / count_all
        std = torch.sqrt(sumsq_all / count_all - mean**2)
        results[variable] = {"mean": float(mean), "std": float(std)}

    for variable in variables_upper:
        pressure = h5.File(files[0], "r")["pressure"][:]
        sum_all = torch.zeros(len(pressure))
        sumsq_all = torch.zeros(len(pressure))
        count_all = torch.zeros(len(pressure))

        for file in files:
            with h5.File(file, "r") as f:
                var = torch.from_numpy(np.array(f[variable]))
                var = F.interpolate(var, size=shape, mode="bilinear")

                mask = torch.isfinite(var)
                var = torch.where(mask, var, float("nan"))

                sum_all += torch.nansum(var, (0, -1, -2))
                sumsq_all += torch.nansum(var**2, (0, -1, -2))
                count_all += mask.sum()

        mean = sum_all / count_all
        std = torch.sqrt(sumsq_all / count_all - mean**2)
        for i, p in enumerate(pressure):
            results[f"{variable}{int(p)}"] = {
                "mean": float(mean[i]),
                "std": float(std[i]),
            }

    output_file = Path(output_dir or env.get("STATS_DIR"), f"stats_{resolution}km.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def _to_4d(x):
    t = torch.from_numpy(x)
    if t.ndim == 4:  # B, Z, H, W
        return t
    elif t.ndim == 3:  # B, H, W
        t = t.unsqueeze(1)
    elif t.ndim == 2:  # H, W
        t = t.unsqueeze(0).unsqueeze(0)

    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", "-r", default=10)
    parser.add_argument(
        "--shape", "-s", type=int, nargs=2, metavar=("H", "W"), default=(90, 90)
    )
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    main(
        resolution=args.resolution,
        shape=tuple(args.shape),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
