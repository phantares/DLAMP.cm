from torch.utils.data import Dataset
import json
import h5py as h5
import numpy as np

from utils import scale_z


class DataDataset(Dataset):
    def __init__(
        self,
        indexes,
        stats_file,
        input_single,
        input_static,
        input_upper,
        target,
        z_input,
        z_target,
    ):
        super().__init__()

        self.indexes = indexes
        with open(stats_file, "r") as f:
            self.stats = json.load(f)

        self.single = input_single
        self.static = input_static
        self.upper = input_upper
        self.target = target

        self.z_input = z_input
        self.z_target = z_target

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index: int):
        file = self.indexes[index]["file"]
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

            return data_single, data_upper, data_target
