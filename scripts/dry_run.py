import torch

torch.set_default_dtype(torch.float64)

from pytorch_lightning.utilities.model_summary import ModelSummary

from model import DirectDownscaling

model = DirectDownscaling(
    global_grid=80,
    resolution_input=10,
    column_km=100,
    resolution_target=2,
    surface_channel=8,
    upper_channel=5,
    output_channel=6,
    crop_number=5,
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
)
summary = ModelSummary(model, max_depth=2)
print(summary)
