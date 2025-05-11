import math

import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gnn.models.model import create_pretrained_model


class LNNP(LightningModule):
    def __init__(self, hparams):
        super(LNNP, self).__init__()

        self.save_hyperparameters(hparams)

        self.model = create_pretrained_model(self.hparams)
        self._reset_losses_dict()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.num_epochs,
            eta_min=self.hparams.lr_min,
            last_epoch=-1,
        )
        return [optimizer], [scheduler]

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")
    
    def loss_weights_schedule(self, num_epochs, type="cosine"):
        curr_epoch = self.trainer.current_epoch
        if type == "cosine":
            return 0.5 * (1 + math.cos(math.pi * curr_epoch / num_epochs))
        elif type == "exp":
            return 1 - (curr_epoch / num_epochs) ** 4
        elif type == "log":
            return 1 - math.log1p(curr_epoch) / math.log1p(num_epochs)
        else:
            raise NotImplementedError(f"Loss weights schedule type {type} not implemented")

    def step(self, batch, stage):
        
        c_loss, atoms_loss, bond_link_loss, bond_class_loss = self(batch)
        
        if self.hparams.arch_chosen == "mlm only":
            c_loss = c_loss * 0
        elif self.hparams.arch_chosen == "contrastive only":
            atoms_loss = atoms_loss * 0
            bond_link_loss = bond_link_loss * 0
            bond_class_loss = bond_class_loss * 0
            
        if self.hparams.use_loss_weights_schedule:
            c_loss_weights = self.loss_weights_schedule(self.hparams.num_epochs, self.hparams.loss_weights_schedule_type)
            c_loss = c_loss_weights * c_loss
        else:
            c_loss_weights = 1.0
        
        loss = atoms_loss  + bond_link_loss + bond_class_loss + c_loss
        self.losses[stage + "_atom"].append(atoms_loss.detach())
        self.losses[stage + "_bond_link"].append(bond_link_loss.detach())
        self.losses[stage + "_bond_class"].append(bond_class_loss.detach())
        self.losses[stage + "_c"].append(c_loss.detach() / c_loss_weights)
        loss_for_logging = (atoms_loss + bond_link_loss + bond_class_loss + c_loss / c_loss_weights).detach()
        self.losses[stage].append(loss_for_logging)

        if stage == "train":
            self.log_dict(
                {
                    "loss_step": loss, 
                    "loss_step_c": c_loss,
                    "loss_step_atom": atoms_loss,
                    "loss_step_bond_link": bond_link_loss,
                    "loss_step_bond_class": bond_class_loss,
                },
                prog_bar=True,
                sync_dist=True,
                logger=False,
            )
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            result_dict["train_loss_atom"] = torch.stack(self.losses["train_atom"]).mean()
            result_dict["train_loss_bond_link"] = torch.stack(self.losses["train_bond_link"]).mean()
            result_dict["train_loss_bond_class"] = torch.stack(self.losses["train_bond_class"]).mean()
            result_dict["train_loss_c"] = torch.stack(self.losses["train_c"]).mean()
            
            result_dict["val_loss_atom"] = torch.stack(self.losses["val_atom"]).mean()
            result_dict["val_loss_bond_link"] = torch.stack(self.losses["val_bond_link"]).mean()
            result_dict["val_loss_bond_class"] = torch.stack(self.losses["val_bond_class"]).mean()
            result_dict["val_loss_c"] = torch.stack(self.losses["val_c"]).mean()

            self.log_dict(result_dict, sync_dist=True)
            if self.hparams.use_loss_weights_schedule:
                self.log("loss_weights", self.loss_weights_schedule(self.hparams.num_epochs, self.hparams.loss_weights_schedule_type))

        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "train_c": [],
            "val_c": [],
            "train_atom": [],
            "val_atom": [],
            "train_bond_link": [],
            "val_bond_link": [],
            "train_bond_class": [],
            "val_bond_class": [],
        }