import torch
import hydra

torch.set_default_dtype(torch.float64)

from pytorch_lightning.utilities.model_summary import ModelSummary


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg):
    if cfg.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32
    torch.set_default_dtype(dtype)

    model = hydra.utils.instantiate(
        cfg.model.system,
        single_channel=len(cfg.dataset.var.input_single)
        + len(cfg.dataset.var.input_static),
        upper_channel=len(cfg.dataset.var.input_upper),
        output_channel=len(cfg.dataset.var.target),
    )
    model = model.to(dtype)

    summary = ModelSummary(model, max_depth=2)
    print(summary)


if __name__ == "__main__":
    main()
