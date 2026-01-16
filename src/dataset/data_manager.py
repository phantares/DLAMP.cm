import os
import lightning as L
import torch
from torch.utils.data import DataLoader

from . import DataIndexer, DataDataset


class DataManager(L.LightningDataModule):

    def __init__(self, **data_configs):
        super().__init__()
        self.save_hyperparameters()

        self.loader_settings = self._get_loader_setting()

    def setup(self, stage: str):
        if stage == "fit" or stage == "test":
            indexer = DataIndexer(
                input_dir=self.hparams.input_dir, **self.hparams.split
            )

            self.train_dataset = DataDataset(
                indexes=indexer.train_index,
                dtype=self.hparams.dtype,
                **self.hparams.res,
                **self.hparams.var,
            )

            self.val_dataset = DataDataset(
                indexes=indexer.val_index,
                dtype=self.hparams.dtype,
                **self.hparams.res,
                **self.hparams.var,
            )

            self.test_dataset = DataDataset(
                indexes=indexer.test_index,
                dtype=self.hparams.dtype,
                **self.hparams.res,
                **self.hparams.var,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            **self.loader_settings,
            **self.hparams.train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            **self.loader_settings,
            **self.hparams.val,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.hparams.test)

    def _get_loader_setting(self):
        gpu_count = torch.cuda.device_count()

        if gpu_count > 1:
            try:
                cpu_count = len(os.sched_getaffinity(0))
            except AttributeError:
                cpu_count = os.cpu_count()

            loader_setting = {
                "num_workers": cpu_count // gpu_count,
                "multiprocessing_context": "spawn",
                "persistent_workers": True,
                "pin_memory": True,
            }

        else:
            loader_setting = {"num_workers": 8}

        return loader_setting
