import lightning as L
from torch.utils.data import DataLoader

from . import DataIndexer, DataDataset


class DataManager(L.LightningDataModule):

    def __init__(self, **data_configs):
        super().__init__()
        self.save_hyperparameters()

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
        return DataLoader(self.train_dataset, shuffle=True, **self.hparams.train)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.hparams.val)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.hparams.test)
