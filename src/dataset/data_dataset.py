from torch.utils.data import Dataset
import json
import h5py as h5
import numpy as np
import torch
from torchvision.transforms.v2 import CenterCrop, Compose, Resize

from utils import scale_z, encode_time


class DataDataset(Dataset):
    def __init__(
        self,
        indexes,
        stats_file,
        global_grid,
        resolution_input,
        resolution_target,
        input_single,
        input_static,
        input_upper,
        target,
        z_input,
        z_target,
        dtype=torch.float64,
    ):
        super().__init__()

        self.dtype = dtype
        self.indexes = indexes
        with open(stats_file, "r") as f:
            self.stats = json.load(f)

        self.single = input_single
        self.static = input_static
        self.upper = input_upper
        self.target = target

        self.z_input = z_input
        self.z_target = z_target

        factor = resolution_input / resolution_target
        self.transform_input = Compose(
            [CenterCrop(int(global_grid * factor)), Resize(global_grid)]
        )
        self.transform_target = Compose([CenterCrop(int(global_grid * factor))])

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index: int):
        file = self.indexes[index]["file"]
        time = self.indexes[index]["time"]
        data_time = encode_time(time, dtype=self.dtype)

        t = self.indexes[index]["index"]
        data_single = []
        data_upper = []
        data_target = []

        with h5.File(file, "r") as f:
            pressure = f["pressure"][:]
            z_up = len(pressure) - 1 - np.searchsorted(pressure[::-1], self.z_input)
            z_tar = len(pressure) - 1 - np.searchsorted(pressure[::-1], self.z_target)

            for variable in self.single:
                data = np.array(f[variable][t,])
                data = scale_z(self.stats, data, variable)
                data_single.append(data)

            for variable in self.static:
                data = np.array(f[variable])
                data = scale_z(self.stats, data, variable)
                data_single.append(data)

            data_single = self._preprocess(data_single, self.transform_input)

            for variable in self.upper:
                data = np.array(f[variable][t,])
                data = data[z_up,]

                for k, z in enumerate(self.z_input):
                    scaled_data = scale_z(
                        self.stats,
                        data[k,],
                        f"{variable}{z}",
                    )
                    data[k,] = scaled_data

                data_upper.append(data)

            data_upper = self._preprocess(data_upper, self.transform_input)

            for variable in self.target:
                data = np.array(f[variable][t,])
                data = data[z_tar,]

                for k, z in enumerate(self.z_target):
                    scaled_data = scale_z(
                        self.stats,
                        data[k,],
                        f"{variable}{z}",
                    )
                    data[k,] = scaled_data

                data_target.append(data)

            data_target = self._preprocess(data_target, self.transform_target)

            return data_single, data_upper, data_time, data_target

    def _preprocess(self, data, transform):
        data = np.array(data)
        data = torch.from_numpy(data)
        data = transform(data)

        return data.to(self.dtype)
