import torch

torch.set_default_dtype(torch.float64)

from pytorch_lightning.utilities.model_summary import ModelSummary

from model import DirectDownscaling

model = DirectDownscaling(
    global_grid=80,
    resolution_input=10,
    resolution_target=2,
    column_km=100,
    crop_number=5,
    single_channel=8,
    upper_channel=5,
    output_channel=5,
    P_mean=0,
    P_std=1,
    sigma_data=0.5,
    z_input=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    z_target=[
        1000,
        975,
        950,
        925,
        900,
        875,
        850,
        825,
        800,
        775,
        750,
        700,
        650,
        600,
        550,
        500,
        450,
        400,
        350,
        300,
        250,
        225,
        200,
        175,
        150,
        125,
        100,
        70,
        50,
        30,
        20,
    ],
    target_var=["qc", "qr", "qi", "qs", "qg"],
    enable_global_encoder=True,
    use_global_token=True,
    use_global_map_embed=True,
    use_global_map_cross_attn=True,
)
summary = ModelSummary(model, max_depth=2)
print(summary)
