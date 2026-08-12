import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import yaml
from dotenv import dotenv_values
from torchvision.transforms.v2 import CenterCrop, Compose, Resize


def main(input_dir):
    env = dotenv_values(".env")

    start_time = datetime.strptime(input_dir.name, "%Y%m%d_%H%M").replace(
        tzinfo=timezone.utc
    )
    with open(input_dir.parent.parent / "var_config.yaml", "r", encoding="utf-8") as f:
        var_configs = yaml.safe_load(f)

    example_file = Path(env.get("INPUT_DIR"), start_time.strftime("%Y%m.h5"))
    output_file = input_dir.parent / f"{input_dir.name}.h5"

    files = sorted(input_dir.glob("surface*.npy"))
    time = [
        (start_time + timedelta(hours=t)).isoformat().encode("utf-8")
        for t in range(len(files))
    ]

    grid_low = np.size(np.load(files[0]), 0)
    grid_high = grid_low * 2
    transform_grid = Compose([CenterCrop(grid_high), Resize(grid_low, antialias=False)])
    transform_input = Compose([CenterCrop(grid_high), Resize(grid_low)])

    vars_static = ["landmask", "terrain"]
    with h5.File(example_file, "r") as f:
        lat = torch.from_numpy(f["latitude"][:])
        lat = transform_grid(lat.unsqueeze(0))

        lon = torch.from_numpy(f["longitude"][:])
        lon = transform_grid(lon.unsqueeze(0))

    with h5.File(example_file, "r") as f_in, h5.File(output_file, "w") as f_out:
        for dim_name, dim in {
            "time": time,
            "pressure": var_configs.pop("pressure"),
            "latitude": lat,
            "longitude": lon,
        }.items():
            if dim_name in f_in:
                dset = f_out.create_dataset(dim_name, data=dim)

                for a_name, a_val in f_in[dim_name].attrs.items():
                    dset.attrs[a_name] = a_val

                f_out[dim_name].make_scale(dim_name)

        for var in vars_static:
            data = torch.from_numpy(f_in[var][:])
            data = data.clamp(min=0.0)
            data = transform_input(data.unsqueeze(0))

            p_ds = f_out.create_dataset(var, data=data, compression="gzip")
            attach_dim(p_ds, *[f_out["latitude"], f_out["longitude"]])

            for a_name, a_val in f_in[var].attrs.items():
                p_ds.attrs[a_name] = a_val

        for source in var_configs:
            for index, var in var_configs[source].items():
                data = []

                files = sorted(input_dir.glob(f"{source}*.npy"))
                for file in files:
                    data.append(np.load(file)[..., index])

                p_ds = f_out.create_dataset(var, data=data, compression="gzip")
                dims = (
                    [f_out["time"], f_out["latitude"], f_out["longitude"]]
                    if source == "surface"
                    else [
                        f_out["time"],
                        f_out["pressure"],
                        f_out["latitude"],
                        f_out["longitude"],
                    ]
                )
                attach_dim(p_ds, *dims)

                for a_name, a_val in f_in[var].attrs.items():
                    p_ds.attrs[a_name] = a_val


def attach_dim(data, *dims):
    for d, dim in enumerate(dims):
        data.dims[d].attach_scale(dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_dir",
        type=str,
        help="Enter input dir.",
    )
    args = parser.parse_args()

    main(Path(args.input_dir))
