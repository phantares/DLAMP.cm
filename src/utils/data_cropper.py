import torch
from torchvision.transforms.v2 import RandomCrop, functional as F


class RandomCropper:
    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def crop(self, surface, upper, crop_number=1):
        surfaces = []
        uppers = []
        cxs = []
        cys = []

        for i in range(surface.size(0)):
            for _ in range(crop_number):
                cropped_surface, cropped_upper, c_x, c_y = self.single_crop(
                    surface[i], upper[i]
                )
                surfaces.append(cropped_surface)
                uppers.append(cropped_upper)
                cxs.append(c_x)
                cys.append(c_y)

        return (
            torch.stack(surfaces),
            torch.stack(uppers),
            torch.tensor(cxs, dtype=surface.dtype),
            torch.tensor(cys, dtype=surface.dtype),
        )

    def single_crop(self, surface, upper):
        i, j, h, w = RandomCrop.get_params(surface, self.crop_shape)

        surface = F.crop(surface, i, j, h, w)
        upper = F.crop(upper, i, j, h, w)

        cx = i + (w - 1) / 2
        cy = j + (h - 1) / 2

        return surface, upper, cx, cy
