import argparse
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize
from utils import find_best_model, get_scaler_map


class OnnxExportWrapper(torch.nn.Module):
    def __init__(
        self, model, mode, input_keys, scaler_map, single_vars, upper_vars, target_vars
    ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.keys = input_keys

        self.scaler = scaler_map
        self.single = single_vars
        self.upper = upper_vars
        self.target = target_vars

    def _scale_var(self, data, vars_name, ndim_after=3, transform=True):
        data = data.clone()
        axis = -(ndim_after + 1)

        for v, var in enumerate(vars_name):
            idx = [slice(None)] * data.dim()
            idx[axis] = slice(v, v + 1)

            scaler = (
                self.scaler[var].transform
                if transform
                else self.scaler[var].inverse_transform
            )

            data[tuple(idx)] = scaler(data[tuple(idx)])

        return data

    def forward(self, *inputs):
        inputs = dict(zip(self.keys, inputs))
        inputs["single"] = self._scale_var(inputs["single"], self.single, ndim_after=2)
        inputs["upper"] = self._scale_var(inputs["upper"], self.upper)

        out = self.model(**inputs)

        result = {}
        for k, v in out.items():
            if k == "regress":
                match self.mode:
                    case "regress":
                        scaled_value = self._scale_var(v, self.target, transform=False)
                        result["regress"] = scaled_value.clamp(min=0)

                    case "norm":
                        C = v.size(-4)
                        result["mu"] = v[..., : C // 2, :, :, :]
                        result["sigma"] = torch.exp(v[..., C // 2 :, :, :, :])
            else:
                result[k] = v

        return result


def main(exp_name):
    with initialize(config_path=f"../experiments/{exp_name}/.hydra", version_base=None):
        cfg = compose(config_name="config")

    dtype = getattr(torch, cfg.dtype, torch.float32)
    torch.set_default_dtype(dtype)

    device = "cpu"

    checkpoint_path = find_best_model(exp_name)
    model_class = hydra.utils.get_class(cfg.model.system._target_)

    model = model_class.load_from_checkpoint(checkpoint_path)
    model.to(device).to(dtype).eval()

    input_keys = list(model.example_input_array.keys())
    dynamic_axes_config = {}
    for k in input_keys:
        dim = len(model.example_input_array[k].shape)

        if k == "column_km":
            dynamic_axes_config[k] = {0: "dim_one"}
        else:
            dynamic_axes_config[k] = {0: "batch_size"}

        if k in ["noise", "sigma", "column_bottom", "column_left"]:
            dynamic_axes_config[k][1] = "crop_number"

        if k in ["single", "upper"]:
            dynamic_axes_config[k][dim - 2] = "h_in"
            dynamic_axes_config[k][dim - 1] = "w_in"
        elif k in ["noise"]:
            dynamic_axes_config[k][dim - 2] = "h_out"
            dynamic_axes_config[k][dim - 1] = "w_out"

    mode = cfg.model.system.output_mode
    upper_vars = cfg.dataset.var.input_upper
    target_vars = cfg.dataset.var.target
    scaler_map = get_scaler_map(
        cfg.dataset.res.stats_file,
        **{var: cfg.dataset.var.z_input for var in upper_vars},
        **{var: cfg.dataset.var.z_target for var in target_vars},
    )

    export_model = (
        OnnxExportWrapper(
            model,
            mode,
            input_keys,
            scaler_map,
            cfg.dataset.var.input_single + cfg.dataset.var.input_static,
            upper_vars,
            target_vars,
        )
        .to(device)
        .to(dtype)
        .eval()
    )

    example_inputs = tuple(model.example_input_array.values())
    with torch.no_grad():
        output_example = export_model(*example_inputs)

    for k, v in output_example.items():
        out_dim = len(v.shape)
        dynamic_axes_config[k] = {
            0: "batch_size",
            1: "crop_number",
            out_dim - 2: "output_h",
            out_dim - 1: "output_w",
        }

    torch.onnx.export(
        export_model,
        example_inputs,
        Path("checkpoints", exp_name, f"{exp_name}.onnx"),
        export_params=True,
        input_names=input_keys,
        output_names=list(output_example.keys()),
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
