import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.gnn_data.circuitnet_deeplayout import CircuitnetDeeplayout
from model.gnn_model.graph_utils_deeplayout import collate_graph_batch


class DInterfaceDeeplayout(pl.LightningDataModule):
    """
    Standalone datamodule for the new DeepLayout-style pipeline.
    Keeps old datamodule untouched.
    """
    def __init__(
        self,
        data_root,
        label_root,
        graph_root,
        batch_size=4,
        num_workers=4,
        graph_ext=".npy",
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.label_root = label_root
        self.graph_root = graph_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.graph_ext = graph_ext
        self.loop = loop

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.trainset = CircuitnetDeeplayout(
                split={"split": "train"},
                data_root=self.data_root,
                label_root=self.label_root,
                graph_root=self.graph_root,
                graph_ext=self.graph_ext,
                loop=self.loop,
            )
            self.valset = CircuitnetDeeplayout(
                split={"split": "val"},
                data_root=self.data_root,
                label_root=self.label_root,
                graph_root=self.graph_root,
                graph_ext=self.graph_ext,
                loop=1,
            )

        if stage == "test" or stage is None:
            self.testset = CircuitnetDeeplayout(
                split={"split": "test"},
                data_root=self.data_root,
                label_root=self.label_root,
                graph_root=self.graph_root,
                graph_ext=self.graph_ext,
                loop=1,
            )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_graph_batch,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_graph_batch,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_graph_batch,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )