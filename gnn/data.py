import warnings

warnings.filterwarnings("ignore")
from os.path import join

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from rdkit.Chem import MCS
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from gnn.datasets import *
from gnn.utils import make_splits, make_stratified_splits


class PretrainedDataModule(LightningDataModule):
    def __init__(self, hparams):
        super(PretrainedDataModule, self).__init__()
        self.hparams.update(hparams.__dict__) if hasattr(
            hparams, "__dict__"
        ) else self.hparams.update(hparams)
        self._saved_dataloaders = dict()
        self.dataset = None

    def prepare_dataset(self):
        self.dataset = MolGraphDataset(
            root=self.hparams["dataset_root"],
            transform=MaskSubgraph(self.hparams["mask_ratio"], self.hparams["seed"]),
        )

        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        test_size = self.hparams["test_size"]

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            test_size,
            False,
            0,
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = store_dataloader and not self.hparams["reload"]
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = PyGDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl


class FinetunedDataModule(LightningDataModule):
    def __init__(self, hparams):
        super(FinetunedDataModule, self).__init__()
        self.hparams.update(hparams.__dict__) if hasattr(
            hparams, "__dict__"
        ) else self.hparams.update(hparams)
        self._saved_dataloaders = dict()
        self.dataset = None

    def prepare_dataset(self):
        assert hasattr(
            self, f"_prepare_{self.hparams['dataset']}_dataset"
        ), f"Dataset {self.hparams['dataset']} not defined"
        dataset_factory = lambda t: getattr(self, f"_prepare_{t}_dataset")()
        self.idx_train, self.idx_val, self.idx_test = dataset_factory(
            self.hparams["dataset"]
        )

        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = store_dataloader and not self.hparams["reload"]
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        dl = PyGDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _prepare_Ontology_dataset(self):
        self.dataset = Ontology(
            root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"]
        )
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        test_size = self.hparams["test_size"]

        idx_train, idx_val, idx_test = make_stratified_splits(
            self.dataset,
            train_size,
            val_size,
            test_size,
            self.hparams["confine_training"],
            self.hparams["num_train_samples"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )
        return idx_train, idx_val, idx_test
    
    def _prepare_Lotus_dataset(self):
        self.dataset = Lotus(root=self.hparams["dataset_root"])
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        test_size = self.hparams["test_size"]

        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            test_size,
            self.hparams["confine_training"],
            self.hparams["confine_ratio"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test
        
    def _prepare_External_dataset(self):
        self.dataset = External(
            root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"]
        )
        val_fold = self.hparams["val_fold"]
        if val_fold == -1:
            idx_train = np.arange(len(self.dataset))
            idx_val = np.where(self.dataset.data.fold == 0)[0]
        else:
            idx_train = np.where(self.dataset.data.fold != val_fold)[0]
            idx_val = np.where(self.dataset.data.fold == val_fold)[0]
        #idx_train = np.arange(len(self.dataset))
        #idx_val = np.random.choice(idx_train, int(0.1 * len(idx_train)), replace=False)
        idx_test = np.array([])
        
        return idx_train, idx_val, idx_test

    def _prepare_Regression_dataset(self):
        self.dataset = NPC(
            root=self.hparams["dataset_root"], dataset_arg=self.hparams["dataset_arg"]
        )
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        test_size = self.hparams["test_size"]

        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            test_size,
            self.hparams["confine_training"],
            self.hparams["confine_ratio"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test
    
    def _prepare_BGC_dataset(self):
        self.dataset = BGC(root=self.hparams["dataset_root"])
        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        test_size = self.hparams["test_size"]

        idx_train, idx_val, idx_test = make_splits(
            len(self.dataset),
            train_size,
            val_size,
            test_size,
            self.hparams["confine_training"],
            self.hparams["confine_ratio"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )

        return idx_train, idx_val, idx_test