import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def project_global_to_roi(global_map, cy, cx, output_size, window_size, global_size):
    device = global_map.device
    k = window_size // 2
    B, C, Z, H, W = global_map.shape

    dy = torch.arange(-k, -k + window_size, device=device)  # (win)
    dx = torch.arange(-k, -k + window_size, device=device)
    YY, XX = torch.meshgrid(dy, dx, indexing="ij")  # (win,win)
    Y = cy[:, None, None] + YY[None]  # (B,win,win)
    X = cx[:, None, None] + XX[None]

    y_norm = (Y + 0.5) / global_size * 2.0 - 1.0
    x_norm = (X + 0.5) / global_size * 2.0 - 1.0
    grid_xy = torch.stack([x_norm, y_norm], dim=-1)  # (B,win,win,2)

    if output_size != window_size:
        grid_xy = rearrange(
            F.interpolate(
                rearrange(grid_xy, "b w1 w2 c -> b c w1 w2"),
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False,
            ),
            "b c w1 w2 -> b w1 w2 c",
        )
    grid_xy = repeat(grid_xy, "b w1 w2 c -> b z w1 w2 c", z=Z)

    z_norm = (torch.arange(Z, device=device) + 0.5) / Z * 2.0 - 1.0
    z_norm = repeat(z_norm, "z -> b z w1 w2 1", b=B, w1=output_size, w2=output_size)
    grid = torch.cat([grid_xy, z_norm], dim=-1)  # (B,Z,win,win,3)

    return F.grid_sample(
        global_map, grid, mode="bilinear", padding_mode="border", align_corners=False
    )  # (B,C_map,Z,out,out)
