import argparse
import os
from os.path import dirname

import numpy as np
import torch
import yaml
from pytorch_lightning.utilities import rank_zero_warn


def train_val_test_split(dset_len, train_size, val_size, test_size, confine_training, confine_ratio, seed):
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."

    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )

    total = train_size + val_size + test_size
    assert (
        dset_len >= total
    ), f"The dataset ({dset_len}) is smaller than the combined split sizes ({total})."

    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int64)
    idxs = np.random.default_rng(seed).permutation(idxs)

    if confine_training:
        confined_train_size = round(train_size * confine_ratio)
        idx_train = idxs[:confined_train_size]
    else:
        idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

# For Classification task with Sparse labels
def stratified_train_val_test_split(labels, train_ratio, val_ratio, test_ratio, confine_training, num_train_samples, seed):
    assert (train_ratio is None) + (val_ratio is None) + (
        test_ratio is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    
    if train_ratio is None:
        train_ratio = 1 - val_ratio - test_ratio
    elif val_ratio is None:
        val_ratio = 1 - train_ratio - test_ratio
    elif test_ratio is None:
        test_ratio = 1 - train_ratio - val_ratio
        
    for ratio in [train_ratio, val_ratio, test_ratio]:
        assert isinstance(ratio, float), "If stratified, train_ratio, val_ratio and test_ratio must be either None or a float"   
        assert 0 < ratio < 1, "If stratified, train_ratio, val_ratio and test_ratio must be between 0 and 1"
    
    unique_labels = np.unique(labels)
    train_indices, val_indices, test_indices = [], [], []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        np.random.default_rng(seed).shuffle(indices)

        train_size = round(len(indices) * train_ratio)
        val_size = round(len(indices) * val_ratio)
        test_size = len(indices) - train_size - val_size
         
        confined_train_size = min(train_size, num_train_samples)
        if confine_training: 
            train = indices[:confined_train_size].tolist()
        else:
            train = indices[:train_size].tolist()
        val = indices[train_size: train_size + val_size].tolist()
        test = indices[train_size + val_size :].tolist()

        train_indices.extend(train)
        val_indices.extend(val)
        test_indices.extend(test)

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)

def make_stratified_splits(dataset, train_ratio, val_ratio, test_ratio, confine_training, num_train_samples, seed, filename=None, splits=None):
    labels = dataset.data.label.numpy()
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = stratified_train_val_test_split(
            labels, train_ratio, val_ratio, test_ratio, confine_training, num_train_samples, seed
        )
    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )
    

def make_splits(
    dataset_len, train_size, val_size, test_size, confine_training, confine_ratio, seed, filename=None, splits=None
):
    if splits is not None:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, confine_training, confine_ratio, seed
        )

    if filename is not None:
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        ckpt = torch.load(values, map_location="cpu")
        config = ckpt["hyper_parameters"]
        for key in config.keys():
            if key not in namespace:
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    os.makedirs(dirname(filename), exist_ok=True)
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def number(text):
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int
    return num_float