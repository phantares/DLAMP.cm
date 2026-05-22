import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from pathlib import Path
from dotenv import dotenv_values
import hydra
from hydra import initialize, compose
import torch
from torchvision.transforms.v2 import CenterCrop
from lightning import Trainer
import h5py as h5
import numpy as np
from itertools import groupby

from dataset import DataIndexer
from utils import find_best_model, get_scaler_map, write_file


def main(exp_name):
    env = dotenv_values(".env")

    with initialize(config_path=f"../experiments/{exp_name}/.hydra", version_base=None):
        cfg = compose(config_name="config")

    dtype = getattr(torch, cfg.dtype, torch.float32)

    experiment_name = cfg.experiment.name
    print(f"Evaluating experiment: {experiment_name}")

    column_km = cfg.dataset.res.global_grid * cfg.dataset.res.resolution_input
    cfg.dataset.res.column_km = column_km
    cfg.dataset.res.crop_number = 1

    checkpoint_path = find_best_model(exp_name)
    model_class = hydra.utils.get_class(cfg.model.system._target_)

    model = model_class.load_from_checkpoint(checkpoint_path, column_km=column_km)
    model.to(torch.float16 if use_mask else dtype)

    logger = hydra.utils.instantiate(cfg.logger, mode="disabled")

    trainer = Trainer(
        logger=logger,
        accelerator="auto",
        devices=1,
        precision="16-true" if use_mask else "16-mixed",
    )

    scaler_map = get_scaler_map(
        cfg.dataset.res.stats_file,
        **{var: cfg.dataset.var.z_target for var in cfg.dataset.var.target},
    )

    global_grid_hr = int(
        cfg.dataset.res.global_grid
        * cfg.dataset.res.resolution_input
        / cfg.dataset.res.resolution_target
    )
    transform_grid = CenterCrop(global_grid_hr)

    test_indexes = DataIndexer(
        Path(env.get("INPUT_DIR")), **cfg.dataset.split
    ).test_index

    for input_file, group in groupby(test_indexes, key=lambda x: x["file"]):
        samples = list(group)

        datamodule = hydra.utils.instantiate(
            cfg.dataset, input_dir=Path(env.get("INPUT_DIR")), dtype=dtype
        )
        datamodule.setup("test")

        datamodule.test_dataset.indexes = samples
        datamodule.setup = lambda stage: None

        model.test_outputs = {k: [] for k in model.test_outputs.keys()}
        trainer.test(model, datamodule=datamodule)

        test_outputs = {
            k: torch.cat(v).to("cpu") for k, v in model.test_outputs.items()
        }

        for c, variable in enumerate(cfg.dataset.var.target):
            invt_pred = (
                scaler_map[variable]
                .inverse_transform(test_outputs["regress"][:, :, c, ...])
                .clamp(min=0)
            )

            test_outputs["regress"][:, :, c, ...] = invt_pred

        test_outputs = {
            k: v.cpu().numpy().reshape(-1, *v.shape[2:])
            for k, v in test_outputs.items()
        }

        indices = [s["index"] for s in samples]
        with h5.File(input_file, "r") as f:
            new_time = f["time"][indices]

            coords = np.stack([f["latitude"][:], f["longitude"][:]])
            transformed_coords = transform_grid(torch.as_tensor(coords)).numpy()
            new_lat, new_lon = transformed_coords[0], transformed_coords[1]

            pressure = f["pressure"][:]
            z_tar = (
                len(pressure)
                - 1
                - np.searchsorted(pressure[::-1], cfg.dataset.var.z_target)
            )

            test_targets = []
            for c, variable in enumerate(cfg.dataset.var.target):
                data = f[variable][indices,]
                data = torch.from_numpy(
                    data[
                        :,
                        z_tar,
                    ]
                )
                data[data < cfg.dataset.var.threshold[variable]] = 0.0
                data = transform_grid(data).numpy()
                test_targets.append(data)
        test_targets = np.stack(test_targets, axis=1)

        new_coords = {
            "time": new_time,
            "pressure": cfg.dataset.var.z_target,
            "latitude": new_lat,
            "longitude": new_lon,
        }

        output_path = Path(env.get("OUTPUT_DIR"), exp_name)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{input_file.stem}.h5"

        write_file(
            input_file,
            output_file,
            new_coords,
            test_outputs,
            test_targets,
            cfg.dataset.var.target,
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
