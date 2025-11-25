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
                input_dir=self.hparams.input_dir,
                val_day=self.hparams.val_day,
                test_day=self.hparams.test_day,
                case_day=self.hparams.case_day,
            )

            self.train_dataset = DataDataset(
                indexes=indexer.train_index,
                input_single=self.hparams.input_single,
                input_static=self.hparams.input_static,
                input_upper=self.hparams.input_upper,
                target=self.hparams.target,
                z_input=self.hparams.z_input,
                z_target=self.hparams.z_target,
            )

            self.val_dataset = DataDataset(
                indexes=indexer.val_index,
                input_single=self.hparams.input_single,
                input_static=self.hparams.input_static,
                input_upper=self.hparams.input_upper,
                target=self.hparams.target,
                z_input=self.hparams.z_input,
                z_target=self.hparams.z_target,
            )

            self.test_dataset = DataDataset(
                indexes=indexer.test_index,
                input_single=self.hparams.input_single,
                input_static=self.hparams.input_static,
                input_upper=self.hparams.input_upper,
                target=self.hparams.target,
                z_input=self.hparams.z_input,
                z_target=self.hparams.z_target,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False)
