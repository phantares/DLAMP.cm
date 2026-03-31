import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import CenterCrop, Compose, Resize

from utils import encode_time


class DataDataset(Dataset):
    def __init__(
        self,
        indexes,
        scaler_map,
        global_grid,
        resolution_input,
        resolution_target,
        column_km,
        crop_number,
        input_single,
        input_static,
        input_upper,
        target,
        threshold,
        z_input,
        z_target,
        dtype=torch.float64,
        mode="train",
    ):
        super().__init__()

        self.mode = mode
        self.handles = {}

        self.column_grid = column_km // resolution_input
        self.target_grid = column_km // resolution_target
        self.crop_number = crop_number

        self.dtype = dtype
        self.indexes = indexes
        self.scaler_map = scaler_map

        self.single = input_single
        self.upper = input_upper
        self.target = target
        self.threshold = {var: threshold.get(var, 0.0) for var in target}

        self.z_input = z_input
        self.z_target = z_target

        self.factor = resolution_input / resolution_target
        global_grid_hr = int(global_grid * self.factor)
        self.transform_grid = Compose(
            [CenterCrop(global_grid_hr), Resize(global_grid, antialias=False)]
        )
        self.transform_input = Compose(
            [CenterCrop(global_grid_hr), Resize(global_grid)]
        )
        self.transform_target = Compose([CenterCrop(global_grid_hr)])

        with h5.File(self.indexes[0]["file"], "r", swmr=True) as f:
            data_static = []
            for variable in input_static:
                data = torch.from_numpy(f[variable][:])
                if variable not in ["longitude", "latitude"]:
                    data = data.clamp(min=0.0)
                data = data.unsqueeze(0).to(self.dtype)

                if variable in ["longitude", "latitude"]:
                    data = self.transform_grid(data)
                else:
                    data = self.transform_input(data)

                data = self.scaler_map[variable].transform(data.squeeze(0))
                data_static.append(data)

            pressure = f["pressure"][:]

            try:
                index_lon = input_static.index("longitude")
                lon = self.scaler_map["longitude"].inverse_transform(
                    data_static[index_lon]
                )

            except ValueError:
                lon = torch.from_numpy(f["longitude"][:])
                lon = lon.clamp(min=0.0)
                lon = lon.unsqueeze(0).to(self.dtype)
                lon = self.transform_grid(lon)
                lon = lon.squeeze(0)

        self.data_static = torch.stack(data_static)
        self.lon = lon.numpy()

        self.z_up = len(pressure) - 1 - np.searchsorted(pressure[::-1], self.z_input)
        self.z_tar = len(pressure) - 1 - np.searchsorted(pressure[::-1], self.z_target)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index: int):
        file = self.indexes[index]["file"]
        time = self.indexes[index]["time"]
        data_time = encode_time(self.lon, time, dtype=self.dtype)

        t = self.indexes[index]["index"]
        data_single = []
        data_upper = []
        data_target = []

        f = self._get_handles(file)

        for variable in self.single:
            data = torch.from_numpy(f[variable][t,])
            if variable not in ["u10", "v10"]:
                data = data.clamp(min=0.0)
            data = data.unsqueeze(0).to(self.dtype)
            data = self.transform_input(data)
            data = self.scaler_map[variable].transform(data.squeeze(0))
            data_single.append(data)

        data_single = torch.stack(data_single)
        data_single = torch.cat((data_single, self.data_static), axis=0)

        for variable in self.upper:
            data = torch.from_numpy(
                f[variable][
                    t,
                    self.z_up,
                ]
            )
            if variable not in ["u", "v", "w"]:
                data = data.clamp(min=0.0)
            data = self.transform_input(data.to(self.dtype))
            data = self.scaler_map[variable].transform(data)
            data_upper.append(data)

        data_upper = torch.stack(data_upper)

        H, W = data_single.shape[-2], data_single.shape[-1]
        if self.mode == "test":
            bottoms = [0]
            lefts = [0]

        else:
            bottoms = np.random.randint(
                0, H - self.column_grid + 1, size=self.crop_number
            )
            lefts = np.random.randint(
                0, W - self.column_grid + 1, size=self.crop_number
            )

        data_target = []
        for variable in self.target:
            data = torch.from_numpy(f[variable][t, self.z_tar])
            data[data < self.threshold[variable]] = 0.0
            data = self.transform_target(data.to(self.dtype))
            data = self.scaler_map[variable].transform(data)

            crop_datas = []
            for n in range(self.crop_number):
                data_bottom = int(bottoms[n] * self.factor)
                data_left = int(lefts[n] * self.factor)

                crop_data = data[
                    ...,
                    data_bottom : data_bottom + self.target_grid,
                    data_left : data_left + self.target_grid,
                ]
                crop_datas.append(crop_data)

            data_target.append(torch.stack(crop_datas))

        data_target = torch.stack(data_target, dim=1)
        bottoms = torch.tensor(bottoms, dtype=self.dtype)
        lefts = torch.tensor(lefts, dtype=self.dtype)

        return data_single, data_upper, data_time, data_target, bottoms, lefts

    def _get_handles(self, file_path):
        if file_path not in self.handles:
            self.handles[file_path] = h5.File(file_path, "r", swmr=True)

        return self.handles[file_path]
