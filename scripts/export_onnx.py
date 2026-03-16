from pathlib import Path
import argparse
import hydra
from hydra import initialize, compose
import torch

from utils import find_best_model


def main(exp_name):
    with initialize(config_path=f"../experiments/{exp_name}/.hydra", version_base=None):
        cfg = compose(config_name="config")

    dtype = getattr(torch, cfg.dtype, torch.float32)
    torch.set_default_dtype(dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = find_best_model(exp_name)
    model_class = hydra.utils.get_class(cfg.model.system._target_)

    column_km = cfg.dataset.res.global_grid * cfg.dataset.res.resolution_input
    model = model_class.load_from_checkpoint(checkpoint_path, column_km=column_km)
    model.to(device).to(dtype).eval()

    input_keys = list(model.example_input_array.keys())
    dynamic_axes_config = {}
    for k in input_keys:
        dim = len(model.example_input_array[k].shape)

        dynamic_axes_config[k] = {0: "batch_size"}

        if k in ["noise", "sigma", "column_bottom", "column_left"]:
            dynamic_axes_config[k][1] = "crop_number"

        if k in ["single", "upper"]:
            dynamic_axes_config[k][dim - 2] = "global_h"
            dynamic_axes_config[k][dim - 1] = "global_w"
        elif k in ["noise"]:
            dynamic_axes_config[k][dim - 2] = "output_h"
            dynamic_axes_config[k][dim - 1] = "output_w"

    with torch.no_grad():
        output_example = model(**model.example_input_array)
        out_dim = len(output_example.shape)

    dynamic_axes_config["output"] = {
        0: "batch_size",
        1: "crop_number",
        out_dim - 2: "output_h",
        out_dim - 1: "output_w",
    }

    model.to_onnx(
        Path("checkpoints", exp_name, f"{exp_name}.onnx"),
        export_params=True,
        input_names=input_keys,
        output_names=["output"],
        dynamic_axes=dynamic_axes_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "exp",
        type=str,
        help="Enter experiment name.",
    )
    args = parser.parse_args()

    main(args.exp)
