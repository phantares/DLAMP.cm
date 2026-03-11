import argparse
from pathlib import Path
from dotenv import dotenv_values
import hydra
from hydra import initialize, compose
from lightning import Trainer
import torch
from torchvision.transforms.v2 import CenterCrop, Compose, Resize
import json
import h5py as h5
import numpy as np
from itertools import groupby

from dataset import DataManager, DataIndexer
from utils import find_best_model, ScalerPipe, write_file


def main(exp_name, wandb_id=None):
    env = dotenv_values(".env")

    with initialize(config_path=f"../experiments/{exp_name}/.hydra", version_base=None):
        cfg = compose(config_name="config")

    if cfg.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.float32
    torch.set_default_dtype(dtype)

    experiment_name = cfg.experiment.name
    print(f"Evaluating experiment: {experiment_name}")

    column_km = cfg.dataset.res.global_grid * cfg.dataset.res.resolution_input
    cfg.dataset.res.column_km = column_km
    cfg.dataset.res.crop_number = 1
    cfg.dataset.res.stats_file = Path(env.get("STATS_DIR")) / cfg.dataset.res.stats_file

    datamodule = DataManager(
        input_dir=Path(env.get("INPUT_DIR")),
        dtype=dtype,
        **cfg.dataset,
    )

    checkpoint_path = find_best_model(exp_name)
    model_class = hydra.utils.get_class(cfg.model.system._target_)

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        column_km=column_km,
    )
    model.to(dtype)

    logger = hydra.utils.instantiate(cfg.logger, id=wandb_id, resume="allow")

    trainer = Trainer(
        logger=logger,
        accelerator="gpu",
        devices=1,
    )

    trainer.test(model, datamodule)

    test_targets = np.concatenate(model.test_targets)
    test_outputs = np.concatenate(model.test_outputs)
    with open(cfg.dataset.res.stats_file) as f:
        stats = json.load(f)

    for c, variable in enumerate(cfg.dataset.var.target):
        for k, z in enumerate(cfg.dataset.var.z_target):
            scaler = ScalerPipe(stats.get(f"{variable}{int(z)}"))
            invt_tar = scaler.inverse_transform(test_targets[:, :, c, k, ...])
            invt_tar[invt_tar < 0] = 0

            invt_pred = scaler.inverse_transform(test_outputs[:, :, c, k, ...])
            invt_pred[invt_pred < 0] = 0

            test_targets[:, :, c, k, ...] = invt_tar
            test_outputs[:, :, c, k, ...] = invt_pred

    test_targets = test_targets.reshape(-1, *test_targets.shape[2:])
    test_outputs = test_outputs.reshape(-1, *test_outputs.shape[2:])

    test_indexes = DataIndexer(
        Path(env.get("INPUT_DIR")), **cfg.dataset.split
    ).test_index

    global_grid_hr = int(
        cfg.dataset.res.global_grid
        * cfg.dataset.res.resolution_input
        / cfg.dataset.res.resolution_target
    )
    transform_grid = Compose(
        [
            CenterCrop(global_grid_hr),
            Resize(cfg.dataset.res.global_grid, antialias=False),
        ]
    )
    current_index = 0
    for input_file, group in groupby(test_indexes, key=lambda x: x["file"]):
        samples = list(group)
        n_samples = len(samples)

        indices = [s["index"] for s in samples]
        with h5.File(input_file, "r") as f:
            new_time = f["time"][indices]

            coords = np.stack([f["latitude"][:], f["longitude"][:]])
            transformed_coords = transform_grid(torch.as_tensor(coords)).numpy()
            new_lat, new_lon = transformed_coords[0], transformed_coords[1]
        print(np.shape(new_lat))

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
            test_outputs[current_index : current_index + n_samples,],
            test_targets[current_index : current_index + n_samples],
            cfg.dataset.var.target,
        )

        current_index += n_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "exp",
        type=str,
        help="Enter experiment name.",
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Enter wandb id.",
    )
    args = parser.parse_args()

    main(args.exp, args.id)
