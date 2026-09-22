import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import yaml
from dotenv import dotenv_values
from torchvision.transforms.v2 import CenterCrop, Compose, Resize
from utils import print_h5_structure


def main(input_dir, file_type):
    env = dotenv_values(".env")

    match file_type:
        case "npy":
            start_time = datetime.strptime(
                input_dir.name,
                "%Y%m%d_%H%M",
            ).replace(tzinfo=timezone.utc)

            example_file = Path(env["INPUT_DIR"], start_time.strftime("%Y%m.h5"))
            output_file = input_dir.parent / f"{input_dir.name}.h5"

            reader = NpyReader(input_dir, example_file)

            write_h5(reader, example_file, output_file)

        case "nc":
            for file in sorted(input_dir.glob("*.nc")):
                example_file = Path(env["INPUT_DIR"], f"{file.stem}.h5")
                output_file = input_dir / f"{file.stem}.h5"

                reader = NcReader(file)
                write_h5(reader, example_file, output_file)


class NpyReader:
    def __init__(self, input_dir, example_file):
        self.input_dir = input_dir

        with open(
            input_dir.parent.parent / "var_config.yaml", "r", encoding="utf-8"
        ) as f:
            config = yaml.safe_load(f)

        self.surface_vars = config["surface"]
        self.upper_vars = config["upper"]
        self.pressure = np.asarray(config["pressure"])

        self.surface_files = sorted(input_dir.glob("surface*.npy"))
        self.upper_files = sorted(input_dir.glob("upper*.npy"))

        start_time = datetime.strptime(
            input_dir.name,
            "%Y%m%d_%H%M",
        ).replace(tzinfo=timezone.utc)

        self.time = [
            (start_time + timedelta(hours=t)).isoformat().encode("utf-8")
            for t in range(len(self.surface_files))
        ]

        with h5.File(example_file, "r") as f:
            lat = torch.from_numpy(f["latitude"][:])
            lon = torch.from_numpy(f["longitude"][:])

        grid_low = np.load(self.surface_files[0]).shape[0]
        grid_high = grid_low * 2
        transform = Compose([CenterCrop(grid_high), Resize(grid_low, antialias=False)])
        self.latitude = transform(lat.unsqueeze(0)).squeeze(0)
        self.longitude = transform(lon.unsqueeze(0)).squeeze(0)

    def get_surface(self, index):
        return np.stack([np.load(file)[..., index] for file in self.surface_files])

    def get_upper(self, index):
        return np.stack([np.load(file)[..., index] for file in self.upper_files])


class NcReader:
    def __init__(self, file):
        self.file = file

        with h5.File(file, "r") as f:
            self.valid = f["valid"][:] > 0
            self.time = [
                datetime.fromtimestamp(t, tz=timezone.utc).isoformat().encode("utf-8")
                for t in f["time"][self.valid]
            ]
            self.pressure = f["level"][:]
            self.latitude = f["lat"][:]
            self.longitude = f["lon"][:]

            with open(
                file.parent.parent / "var_config.yaml", "r", encoding="utf-8"
            ) as config_file:
                config = yaml.safe_load(config_file)

            surface_vars = [
                v.decode("utf-8") if isinstance(v, bytes) else str(v)
                for v in list(f["surface_var"][:])
            ]
            self.surface_vars = {
                i: config["surface"][name] for i, name in enumerate(surface_vars)
            }

            upper_vars = [
                v.decode("utf-8") if isinstance(v, bytes) else str(v)
                for v in list(f["upper_var"][:])
            ]
            self.upper_vars = {
                i: config["upper"][name] for i, name in enumerate(upper_vars)
            }

    def get_surface(self, index):
        with h5.File(self.file, "r") as f:
            return f["surface"][self.valid, ..., index]

    def get_upper(self, index):
        with h5.File(self.file, "r") as f:
            return f["upper"][self.valid, ..., index]


def write_h5(reader, example_file, output_file):
    vars_static = ["landmask", "terrain"]

    with h5.File(example_file, "r") as f_in, h5.File(output_file, "w") as f_out:
        for dim_name, dim in {
            "time": reader.time,
            "pressure": reader.pressure,
            "latitude": reader.latitude,
            "longitude": reader.longitude,
        }.items():
            dset = f_out.create_dataset(dim_name, data=dim)

            if dim_name in f_in:
                for name, value in f_in[dim_name].attrs.items():
                    dset.attrs[name] = value

            dset.make_scale(dim_name)

        grid_low = reader.longitude.shape[0]
        grid_high = grid_low * 2
        transform_input = Compose([CenterCrop(grid_high), Resize(grid_low)])
        for var in vars_static:
            data = torch.from_numpy(f_in[var][:])
            data = data.clamp(min=0.0)
            data = transform_input(data.unsqueeze(0)).squeeze(0)

            p_ds = f_out.create_dataset(var, data=data, compression="gzip")
            attach_dim(p_ds, *[f_out["latitude"], f_out["longitude"]])

            for a_name, a_val in f_in[var].attrs.items():
                p_ds.attrs[a_name] = a_val

        for index, var in reader.surface_vars.items():
            data = reader.get_surface(index)

            p_ds = f_out.create_dataset(var, data=data, compression="gzip")

            attach_dim(p_ds, f_out["time"], f_out["latitude"], f_out["longitude"])

            if var in f_in:
                for name, value in f_in[var].attrs.items():
                    p_ds.attrs[name] = value

        for index, var in reader.upper_vars.items():
            data = reader.get_upper(index)

            p_ds = f_out.create_dataset(var, data=data, compression="gzip")

            attach_dim(
                p_ds,
                f_out["time"],
                f_out["pressure"],
                f_out["latitude"],
                f_out["longitude"],
            )

            if var in f_in:
                for name, value in f_in[var].attrs.items():
                    p_ds.attrs[name] = value

    print_h5_structure(output_file)
    print(f"File saved : {output_file}")


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
    parser.add_argument(
        "file_type",
        type=str,
        help="Enter input file type.",
    )
    args = parser.parse_args()

    main(Path(args.input_dir), args.file_type)
