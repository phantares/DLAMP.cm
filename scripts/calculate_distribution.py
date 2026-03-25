from pathlib import Path
from dotenv import dotenv_values
import argparse
import h5py as h5
import numpy as np
import torch

BINS = [
    0,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
]
variables = ["qi", "qs", "qg", "qc", "qr"]


def main(input_dir=None, output_dir=None, cloud_threshold=1e-8):
    env = dotenv_values(".env")

    input_dir = Path(input_dir or env.get("INPUT_DIR"))
    files = [h5.File(file, "r") for file in sorted(input_dir.glob("*.h5"))]
    output_dir = Path(output_dir or env.get("STATS_DIR"))

    pressure = files[0]["pressure"]

    for variable in variables:
        counts = []

        for k, p in enumerate(pressure):
            print(f"{variable}{int(p)}")
            vars = []

            for file in files:
                var = np.array(
                    file[variable][
                        :,
                        k,
                    ]
                )
                var[var < cloud_threshold] = 0
                var = _to_3d(var)

                vars.append(var)

            vars = torch.concatenate(vars, axis=0)

            count, _ = np.histogram(vars.numpy(), bins=BINS)
            counts.append(count)

        output_file = output_dir / f"{variable}_distribution"
        np.save(output_file, counts)


def _to_3d(x):
    t = torch.from_numpy(x)
    if t.ndim == 3:  # B, H, W
        return t
    elif t.ndim == 2:  # H, W
        t = t.unsqueeze(0)

    return t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cloud_threshold", default=1e-8, type=float)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cloud_threshold=args.cloud_threshold,
    )
