import torch
from torchvision.transforms.v2 import RandomCrop, functional as F


class RandomCropper:
    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def crop(self, data, crop_number=1):
        device = data.device
        dtype = data.dtype

        datas = []
        tops = []
        lefts = []

        for i in range(data.size(0)):
            for _ in range(crop_number):
                cropped_data, top, left = self.single_crop(data[i])

                datas.append(cropped_data)
                tops.append(top)
                lefts.append(left)

        return (
            torch.stack(datas).to(dtype),
            torch.tensor(tops, dtype=dtype, device=device),
            torch.tensor(lefts, dtype=dtype, device=device),
        )

    def single_crop(self, data):
        i, j, h, w = RandomCrop.get_params(data, self.crop_shape)
        data = F.crop(data, i, j, h, w)

        return data, i, j
