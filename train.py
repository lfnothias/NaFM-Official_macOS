import argparse
import logging
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from gnn.data import PretrainedDataModule, FinetunedDataModule
from gnn.pre_module import LNNP as PretrainedLNNP
from gnn.tune_module import LNNP as FinetunedLNNP
from gnn.utils import LoadFromFile, number, save_argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    )  # keep second

    # training settings
    parser.add_argument("--num-epochs", default=300, type=int, help="number of epochs")
    parser.add_argument("--lr", default=1.0e-4, type=float, help="learning rate")
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1.0e-5,
        help="Minimum learning rate before early stop",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1.0e-5, help="Weight decay strength"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )

    parser.add_argument(
        "--dataset-root",
        default="coconut_data",
        type=str,
        help="Data storage directory",
    )
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="Mask ratio")
    
    # Finetuned dataset specific
    parser.add_argument(
        "--dataset", 
        default=None, 
        type=str, 
        help="Finetuned Dataset name",
        choices=['Lotus','Ontology','Regression', 'External','BGC'],
    )
    
    parser.add_argument(
        "--dataset-arg",
        default=None,
        type=str,
        help="Ontology/Regression Dataset argument",          
    )
    
    parser.add_argument(
        "--confine-training",
        action=argparse.BooleanOptionalAction,
        help="Confine training to a subset of the training set",
    )
    
    parser.add_argument(
        "--num-train-samples",
        type=int,
        default=4,
        help="Number of samples to confine training set to",
    )
    
    parser.add_argument(
        "--confine-ratio",
        type=float,
        default=0.2,
        help="Number of samples to confine training set to",
    )
    parser.add_argument(
        "--val-fold",
        type=int,
        default= 0,
        help="Fold to use as validation set",
    )
    # dataloader specific
    parser.add_argument(
        "--reload", type=int, default=0, help="Reload dataloaders every n epoch"
    )
    parser.add_argument("--batch-size", default=512, type=int, help="batch size")
    parser.add_argument(
        "--inference-batch-size",
        default=None,
        type=int,
        help="Batchsize for validation and tests.",
    )
    parser.add_argument(
        "--splits", default=None, help="Npz with splits idx_train, idx_val, idx_test"
    )
    parser.add_argument(
        "--train-size",
        type=number,
        default=0.8,
        help="Percentage/number of samples in training set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--val-size",
        type=number,
        default=0.1,
        help="Percentage/number of samples in validation set (None to use all remaining samples)",
    ) 
    parser.add_argument(
        "--test-size",
        type=number,
        default=0.1,
        help="Percentage/number of samples in test set (None to use all remaining samples)",
    ) 
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of workers for data prefetch"
    )
    
    # architectural specific
    parser.add_argument("--emb-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument(
        "--num-layer", type=int, default=8, help="Number of layers in the model"
    )
    parser.add_argument("--feat-dim", type=int, default=1024, help="Feature dimension")
    parser.add_argument("--drop-ratio", type=float, default=0.1, help="Dropout ratio")
    parser.add_argument(
        "--linear-drop-ratio",
        type=float,
        default=0.3,
        help="Dropout ratio for linear layers",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gin",
        choices=["gin", "gcn", "gat", "graphsage"],
        help="GNN type",
    )
    parser.add_argument(
        "--freeze",
        action=argparse.BooleanOptionalAction,
        help="Freeze the representation model",
    )
    parser.add_argument(
        "--include-target",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Include target in the input",
    )
    parser.add_argument(
        "--screen-coconut-weights",
        default=1,
        type=float,
        help="Screen coconut weights",
    )
    parser.add_argument(
        "--use-loss-weights-schedule",
        action=argparse.BooleanOptionalAction,
        help="Use loss weights schedule",
    )
    parser.add_argument(
        "--loss-weights-schedule-type",
        type=str,
        default="cosine",
        help="Loss weights schedule type",
    )
    parser.add_argument(
        "--arch-chosen",
        type=str,
        default="default",
        choices=["default", "mlm only", "contrastive only"],
        help="Architecture chosen", 
    )
    
    # other specific
    parser.add_argument(
        "--ngpus",
        type=int,
        default=-1,
        help="Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus",
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=[16, 32],
        help="Floating point precision",
    )
    parser.add_argument("--log-dir", type=str, default="log", help="Log directory")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--distributed-backend", default="ddp", help="Distributed backend"
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        help='Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")',
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save interval, one save per n epochs (default: 10)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pretrain",
        choices=["pretrain", "finetune", "ecfp"],
        help="Task to train",
    )
    args = parser.parse_args()

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    save_argparse(args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"])

    return args


def main():
    args = get_args()
    pl.seed_everything(args.seed, workers=True)

    # initialize data module
    data = FinetunedDataModule(args) if args.task != "pretrain" else PretrainedDataModule(args)
    data.prepare_dataset()

    # initialize lightning module
    num_classes = data.dataset.num_class if args.dataset != "Regression" and args.task != "pretrain" else 1
    model = FinetunedLNNP(args, num_classes) if args.task != "pretrain" else PretrainedLNNP(args)
    monitor = "val_auprc" if (args.dataset == "Ontology" or args.dataset == "BGC") else "val_loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.log_dir,
        monitor=monitor,
        save_top_k=100,
        save_last=True,
        every_n_epochs=args.save_interval,
        filename="{epoch}-{" + monitor + ":.4f}" if args.dataset != "External" else "{epoch}-{val_loss:.4f}-{ef1:.4f}-{ef5:.4f}-{ef10:.4f}",
        mode="max" if monitor == "val_auprc" else "min",
    )
    
    callbacks = [checkpoint_callback]

    if args.task != "pretrain":
        early_stopping = EarlyStopping(
            monitor, 
            patience=args.early_stopping_patience, 
            mode="max" if monitor == "val_auprc" else "min"
        )
        callbacks.append(early_stopping)

    tb_logger = TensorBoardLogger(
        args.log_dir, name="tensorbord", version="", default_hp_metric=False
    )
    csv_logger = CSVLogger(args.log_dir, name="", version="")

    num_devices = args.ngpus if args.accelerator != "cpu" else 1

    # before Trainer: decide whether to use DDP
    ddp_plugin = DDPStrategy(find_unused_parameters=False) if args.accelerator != "cpu" else None

    # build common Trainer kwargs
    trainer_kwargs = {
        "log_every_n_steps": 50,
        "max_epochs":           args.num_epochs,
        "accelerator":          args.accelerator,
        # CPUAccelerator expects an int > 0; for GPU this is # of GPUs
        "devices":              args.ngpus if args.accelerator != "cpu" else 1,
        "num_nodes":            args.num_nodes,
        "default_root_dir":     args.log_dir,
        "callbacks":            callbacks,
        "logger":               [tb_logger, csv_logger],
        "reload_dataloaders_every_n_epochs": args.reload,
        "precision":            args.precision,
        "enable_progress_bar":  True,
        "gradient_clip_val":    1.0,
        "gradient_clip_algorithm": "norm",
    }

    # only insert strategy when on GPU
    if ddp_plugin is not None:
        trainer_kwargs["strategy"] = ddp_plugin

    # instantiate the Trainer with your assembled kwargs
    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, datamodule=data)

if __name__ == "__main__":
    main()
