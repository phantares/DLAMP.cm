import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms.v2 import RandomCrop, functional as Ftrans


def crop_column(
    data,
    column_bottom,
    column_left,
    window_shape,
    output_shape=None,
    mode="bilinear",
    align_corners=False,
):
    shape_orig = data.shape
    has_z = len(shape_orig) == 5

    if has_z:
        data = rearrange(data, "b c z h w -> b (c z) h w")

    B, N = column_bottom.shape
    H_in, W_in = data.shape[-2:]
    H_win, W_win = window_shape
    H_out, W_out = output_shape or window_shape

    yy = torch.arange(H_out[0]).to(data.dtype).to(data.device)
    xx = torch.arange(W_out[0]).to(data.dtype).to(data.device)
    yy = yy * ((H_win - 1) / (H_out - 1)) if H_out > 1 else yy
    xx = xx * ((W_win - 1) / (W_out - 1)) if W_out > 1 else xx
    YY, XX = torch.meshgrid(yy, xx, indexing="ij")
    Y = YY[None, None, ...] + column_bottom[:, :, None, None]
    X = XX[None, None, ...] + column_left[:, :, None, None]

    if align_corners:
        norm_x = (2.0 * X / (W_in - 1)) - 1.0
        norm_y = (2.0 * Y / (H_in - 1)) - 1.0
    else:
        norm_x = (2.0 * (X + 0.5) / W_in) - 1.0
        norm_y = (2.0 * (Y + 0.5) / H_in) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1)
    grid = rearrange(grid, "b n h w c -> b (n h) w c")

    cropped_column = F.grid_sample(
        data,
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=align_corners,
    )

    if has_z:
        cropped_column = rearrange(
            cropped_column,
            "b (c z) (n h) w -> b n c z h w",
            c=shape_orig[1],
            z=shape_orig[-3],
            n=N,
        )
    else:
        cropped_column = rearrange(cropped_column, "b c (n h) w -> b n c h w", n=N)

    return cropped_column


class RandomCropper:
    def __init__(self, crop_shape):
        self.crop_shape = crop_shape

    def crop(self, data, crop_number=1):
        device = data.device
        dtype = data.dtype

        datas = []
        bottoms = []
        lefts = []

        for i in range(data.size(0)):
            for _ in range(crop_number):
                cropped_data, top, left = self.single_crop(data[i])

                datas.append(cropped_data)
                bottoms.append(top)
                lefts.append(left)

        return (
            torch.stack(datas).to(dtype),
            torch.tensor(bottoms, dtype=dtype, device=device),
            torch.tensor(lefts, dtype=dtype, device=device),
        )

    def single_crop(self, data):
        i, j, h, w = RandomCrop.get_params(data, self.crop_shape)
        data = Ftrans.crop(data, i, j, h, w)

        return data, i, j
