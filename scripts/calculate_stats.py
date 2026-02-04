from pathlib import Path
from dotenv import dotenv_values
import argparse
import h5py as h5
import numpy as np
import json
import torch
from torchvision.transforms.v2 import Resize


variables_constant = ["terrain", "latitude", "longitude"]
variables_single = ["sst", "psfc", "t2m", "q2m", "u10", "v10", "olr", "sw"]
variables_upper = ["z", "u", "v", "w", "t", "qv", "qi", "qs", "qg", "qc", "qr"]
variables_cloud = ["qi", "qs", "qg", "qc", "qr"]


def main(resolution, shape, input_dir=None, output_dir=None, cloud_threshold=1e-6):
    env = dotenv_values(".env")

    dir = Path(input_dir or env.get("INPUT_DIR"))
    files = sorted(dir.glob("*.h5"))
    results = {}

    for variable in variables_constant:
        print(variable)

        with h5.File(files[0], "r") as f:
            var = np.array(f[variable][:])
            var[var < 0] = 0
            var = _to_3d(var)

            if variable in ["longitude", "latitude"]:
                resize = Resize(shape, antialias=False)
            else:
                resize = Resize(shape)
            var = resize(var)

            results[variable] = {
                "scaling": "minmax",
                "mean": float(torch.mean(var)),
                "std": float(torch.std(var)),
                "min": float(torch.min(var)),
                "max": float(torch.max(var)),
            }

    for variable in variables_single:
        print(variable)

        vars = []
        for file in files:
            with h5.File(file, "r") as f:
                var = np.array(f[variable][:])
                if variable not in ["u10", "v10"]:
                    var[var < 0] = 0
                var = _to_3d(var)
                var = Resize(shape)(var)

                vars.append(var)

        vars = torch.concatenate(vars, axis=0)

        results[variable] = {
            "scaling": "minmax",
            "mean": float(torch.mean(vars)),
            "std": float(torch.std(vars)),
            "min": float(torch.min(vars)),
            "max": float(torch.max(vars)),
        }

    pressure = h5.File(files[0], "r")["pressure"][:]
    for variable in variables_upper:

        for k, p in enumerate(pressure):
            print(f"{variable}{int(p)}")
            vars = []

            for file in files:
                with h5.File(file, "r") as f:
                    var = np.array(
                        f[variable][
                            :,
                            k,
                        ]
                    )
                    if variable not in ["u", "v", "w"]:
                        var[var < 0] = 0
                    if variable in variables_cloud:
                        var[var < cloud_threshold] = 0
                    var = _to_3d(var)
                    var = Resize(shape)(var)

                    vars.append(var)

            vars = torch.concatenate(vars, axis=0)

            stats = {
                "scaling": "minmax",
                "mean": float(torch.mean(vars)),
                "std": float(torch.std(vars)),
                "min": float(torch.min(vars)),
                "max": float(torch.max(vars)),
            }

            if variable in variables_cloud:
                pr10 = torch.quantile(vars[vars > 0], 0.1)
                log1 = torch.log10(vars / pr10 + 1)
                pr10_1 = torch.quantile(log1[log1 > 0], 0.1)
                log2 = torch.log10(vars / pr10_1 + 1)

                stats["pr10"] = float(pr10)
                stats["min_log1"] = float(torch.min(log1))
                stats["max_log1"] = float(torch.max(log1))
                stats["pr10_log1"] = float(pr10_1)
                stats["min_log2"] = float(torch.min(log2))
                stats["max_log2"] = float(torch.max(log2))

            results[f"{variable}{int(p)}"] = stats

    output_file = Path(output_dir or env.get("STATS_DIR"), f"stats_{resolution}km.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def _to_3d(x):
    t = torch.from_numpy(x)
    if t.ndim == 3:  # B, H, W
        return t
    elif t.ndim == 2:  # H, W
        t = t.unsqueeze(0)

    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", "-r", default=10)
    parser.add_argument(
        "--shape", "-s", type=int, nargs=2, metavar=("H", "W"), default=(90, 90)
    )
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cloud_threshold", default=1e-6)
    args = parser.parse_args()

    main(
        resolution=args.resolution,
        shape=tuple(args.shape),
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cloud_threshold=args.cloud_threshold,
    )
