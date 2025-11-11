def crop_center(x, ref):
    _, _, _, H, W = x.shape
    _, _, _, Hr, Wr = ref.shape

    y0 = max(0, (H - Hr) // 2)
    x0 = max(0, (W - Wr) // 2)

    return x[:, :, :, y0 : y0 + Hr, x0 : x0 + Wr]
