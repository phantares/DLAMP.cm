import torch


def interpolate_z(data, z_source, z_target):
    Zs = data.size(-3)
    device = data.device

    z_src = torch.as_tensor(z_source, dtype=data.dtype, device=device)
    z_tgt = torch.as_tensor(z_target, dtype=data.dtype, device=device)

    zs = z_src.log()
    zt = z_tgt.log()

    if zs[-1] < zs[0]:  # descending
        zs_asc = torch.flip(zs, dims=[0])
        data_asc = torch.flip(data, dims=[-3])
    else:
        zs_asc = zs
        data_asc = data

    layer_below = torch.searchsorted(zs_asc, zt)
    layer_above = layer_below - 1

    layer_below = layer_below.clamp(1, Zs - 1)
    layer_above = layer_above.clamp(0, Zs - 2)

    z0 = zs_asc[layer_above]  # (Zt,)
    z1 = zs_asc[layer_below]  # (Zt,)
    ratio = (zt - z0) / (z1 - z0)  # (Zt,)

    data_below = data_asc.index_select(-3, layer_below)  # (B,C,Zt,H,W)
    data_above = data_asc.index_select(-3, layer_above)  # (B,C,Zt,H,W)
    ratio = ratio.view(1, 1, -1, 1, 1).to(data.dtype)
    output = ratio * data_below + (1 - ratio) * data_above  # (B,C,Zt,H,W)

    return output
