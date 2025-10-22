import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from gnn.models.model import create_finetuned_model, create_ecfp4_model
import numpy as np

def calculate_ef1_percent(y_true, y_pred, threshold):
    top_percent_threshold = np.quantile(y_pred, threshold)
    total_molecules = len(y_true)
    active_molecules = len(y_true[y_true == 1])
    print(f'total_molecules:{total_molecules}, active_molecules: {active_molecules}')
    top_percent_active = len(y_true[(y_pred >= top_percent_threshold) & (y_true == 1)])
    top_percent_total = len(y_true[y_pred >= top_percent_threshold])
    print(f'top{100*(1-threshold):.1f}%active molecules:{top_percent_active}, top{100*(1-threshold):.1f}%total molecules: {top_percent_total}')
    top_percent = top_percent_active / top_percent_total if top_percent_total > 0 else 0
    total_percent = active_molecules / total_molecules if total_molecules > 0 else 0
    return top_percent / total_percent if total_percent > 0 else 0

class LNNP(LightningModule):
    def __init__(self, hparams, num_classes, is_inference=False):
        super(LNNP, self).__init__()
        
        self.num_classes = num_classes
        self.save_hyperparameters(hparams)
        if self.hparams['task'] == 'ecfp':
            self.model = create_ecfp4_model(self.hparams, num_classes)
        else:
            self.model = create_finetuned_model(
                self.hparams, num_classes,
                freeze=self.hparams['freeze'],
                is_inference=is_inference
            )
        
        self._reset_losses_dict()
        
        if self.hparams["dataset"] != 'Regression' and self.hparams['dataset'] != 'BGC':
            if self.hparams['dataset'] == 'External':
                self.loss = torch.nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss = torch.nn.CrossEntropyLoss()    

        elif self.hparams['dataset'] == 'BGC':
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = torch.nn.MSELoss()
            
        self.y_true_records = {"val":[]}
        self.y_pred_records = {"val":[]}
            
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
        if self.hparams['task'] == 'ecfp':
            return self.model(data.ecfp.reshape(-1, 2048))
        else:
            return self.model(data)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        if self.hparams['dataset'] != 'Regression':
            if self.hparams['dataset'] == 'BGC':
                y_true = batch.label.detach().cpu().numpy()
                y_pred = self(batch).detach().cpu().numpy()
                y_pred = torch.sigmoid(torch.tensor(y_pred)).numpy()
    
                self.y_true_records["val"].append(y_true)
                self.y_pred_records["val"].append(y_pred)
    
                return self.step(batch, "val")  
            else:
                y_true = batch.label.detach().cpu().numpy()
                y_true = label_binarize(y_true, classes=[i for i in range(self.num_classes)])
                y_pred = self(batch).detach().cpu().numpy()
        
                self.y_true_records["val"].append(y_true)
                self.y_pred_records["val"].append(y_pred)

                return self.step(batch, "val")
        else:
            return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def step(self, batch, stage):
        prob = self(batch)
        loss = self.loss(prob, batch.label)
        if self.hparams['dataset'] == 'External':
            weights = torch.where(batch.is_coconut.to(torch.bool), self.hparams['screen_coconut_weights'], 1)
            loss = (loss * weights).mean()
        self.losses[stage].append(loss.detach())
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            if self.hparams['dataset'] != 'Regression' and self.hparams['dataset'] != 'External':
                y_true = np.concatenate(self.y_true_records["val"])
                y_pred = np.concatenate(self.y_pred_records["val"])
                auprc, count = np.zeros(self.num_classes), 0
                for i in range(self.num_classes):
                # Only compute AUC for valid classes
                    if len(np.unique(y_true[:, i])) > 1:
                        auprc[i] = average_precision_score(y_true[:,i], y_pred[:, i])
                        count += 1

                auprc = np.sum(auprc) / count if count > 0 else float('nan')
            
                result_dict = {
                    "epoch": float(self.current_epoch),
                    "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                    "train_loss": torch.stack(self.losses["train"]).mean(),
                    "val_loss": torch.stack(self.losses["val"]).mean(),
                    "val_auprc": auprc,
                }

                self.log_dict(result_dict, sync_dist=True)
            
            elif self.hparams['dataset'] == 'External':
                y_true = np.concatenate(self.y_true_records["val"]).ravel()
                y_pred = np.concatenate(self.y_pred_records["val"])
                y_pred_positive_probs = y_pred[:, 1]
                ef1 = calculate_ef1_percent(y_true, y_pred_positive_probs, 0.99)
                ef5 = calculate_ef1_percent(y_true, y_pred_positive_probs, 0.95)
                ef10 = calculate_ef1_percent(y_true, y_pred_positive_probs, 0.90)
                result_dict = {
                    "epoch": float(self.current_epoch),
                    "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                    "train_loss": torch.stack(self.losses["train"]).mean(),
                    "val_loss": torch.stack(self.losses["val"]).mean(),
                    "ef1": ef1,
                    "ef5": ef5,
                    "ef10": ef10,
                } 
                
                self.log_dict(result_dict, sync_dist=True)
                
            else:
                result_dict = {
                    "epoch": float(self.current_epoch),
                    "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                    "train_loss": torch.stack(self.losses["train"]).mean(),
                    "val_loss": torch.stack(self.losses["val"]).mean(),
                }

                self.log_dict(result_dict, sync_dist=True)
                
        self._reset_losses_dict()
        self._reset_records_dict()

    def on_test_epoch_end(self):
        result_dict = {}

        if len(self.losses["test"]) > 0:
            result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()
            
        self.log_dict(result_dict, sync_dist=True)

        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
        }
        
    def _reset_records_dict(self):
        self.y_true_records = {"val": []}
        self.y_pred_records = {"val": []}