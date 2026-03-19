import os
import lightning as L
import torch
from torch.utils.data import DataLoader

from . import DataIndexer, DataDataset
from utils import get_scaler_map, ScalerPipe, StackedScalerPipe


class DataManager(L.LightningDataModule):

    def __init__(self, **data_configs):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str):
        stats_file = self.hparams.res.stats_file
        scaler_map = get_scaler_map(stats_file)

        for variable in self.hparams.var.input_upper:
            pipelines = [
                scaler_map.get(f"{variable}{int(z)}", ScalerPipe(None))
                for z in self.hparams.var.z_input
            ]
            scaler_map[variable] = StackedScalerPipe(pipelines)

        for variable in self.hparams.var.target:
            pipelines = [
                scaler_map.get(f"{variable}{int(z)}", ScalerPipe(None))
                for z in self.hparams.var.z_target
            ]
            scaler_map[variable] = StackedScalerPipe(pipelines)

        res_config = {k: v for k, v in self.hparams.res.items() if k != "stats_file"}

        if stage == "fit" or stage == "test":
            indexer = DataIndexer(
                input_dir=self.hparams.input_dir, **self.hparams.split
            )

            self.train_dataset = DataDataset(
                indexes=indexer.train_index,
                scaler_map=scaler_map,
                dtype=self.hparams.dtype,
                **res_config,
                **self.hparams.var,
            )

            self.val_dataset = DataDataset(
                indexes=indexer.val_index,
                scaler_map=scaler_map,
                dtype=self.hparams.dtype,
                **res_config,
                **self.hparams.var,
            )

            self.test_dataset = DataDataset(
                mode="test",
                indexes=indexer.test_index,
                scaler_map=scaler_map,
                dtype=self.hparams.dtype,
                **res_config,
                **self.hparams.var,
            )

    def train_dataloader(self):
        loader_settings = self._get_loader_setting("fit")

        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **loader_settings,
            **self.hparams.train,
        )

    def val_dataloader(self):
        loader_settings = self._get_loader_setting("fit")

        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **loader_settings,
            **self.hparams.val,
        )

    def test_dataloader(self):
        loader_settings = self._get_loader_setting("test")

        return DataLoader(
            self.test_dataset,
            shuffle=False,
            **loader_settings,
            **self.hparams.test,
        )

    def _get_loader_setting(self, stage):
        gpu_count = torch.cuda.device_count()

        try:
            cpu_count = len(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count()

        if gpu_count > 1:
            if stage == "fit":
                num_workers = cpu_count // gpu_count
            else:
                num_workers = cpu_count - 2

        else:
            num_workers = min(8, cpu_count)

        return {
            "num_workers": num_workers,
            "multiprocessing_context": "spawn",
            "persistent_workers": True,
            "pin_memory": True,
        }
