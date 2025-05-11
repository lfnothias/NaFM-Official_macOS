import re

import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data

from gnn.models.gnn import PretrainedGNN, FinetunedGNN, Linear_Block
from gnn.models.utils import NTXentLoss, MLMLoss
import torch.nn.functional as F
from collections import OrderedDict

def create_pretrained_model(args):
    gnn_args = dict(
        num_layer=args["num_layer"],
        emb_dim=args["emb_dim"],
        drop_ratio=args["drop_ratio"],
        gnn_type=args["gnn_type"],
        feat_dim=args["feat_dim"],
    )

    pretrain_gnn = PretrainedGNN(**gnn_args)
    contrastive_loss = NTXentLoss(
        temperature=args["temperature"], use_cosine_similarity=True
    )
    mlm_loss = MLMLoss()

    model = PretrainedModel(
        pretrain_gnn,
        contrastive_loss,
        mlm_loss=mlm_loss,
    )
    return model


def create_finetuned_model(args, num_classes, freeze=True):
    gnn_args = dict(
        num_layer=args["num_layer"],
        emb_dim=args["emb_dim"],
        drop_ratio=args["drop_ratio"],
        linear_drop_ratio=args["linear_drop_ratio"],
        gnn_type=args["gnn_type"],
    )
    if args['pretrained_path'] is None:
        print("Train from scratch")
        model = FinetunedGNN(**gnn_args, num_classes=num_classes)
    else:
        print("Load pretrained model")
        pretrained_model = load_pretrained_model(args["pretrained_path"])
        representation_model = pretrained_model.pretrain_gnn.representation_model
        
        if freeze:
            print("Freeze the representation model")
            for param in representation_model.parameters():
                param.requires_grad = False
            print("Disable dropout in finetuning")
            representation_model = representation_model.eval()
         
        model = FinetunedGNN(**gnn_args, num_classes=num_classes, include_target=args.include_target)
        model.representation_model = representation_model

    return model


def create_ecfp4_model(args, num_classes):
    return Linear_Block(
        input_size=args["emb_dim"],
        hidden_size=args["emb_dim"] // 2,
        num_classes=num_classes,
        dropout_rate=args["linear_drop_ratio"],
    )

def load_pretrained_model(filepath, args=None, device="cpu", **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if key not in args:
            rank_zero_warn(f"Unknown hyperparameter: {key}={value}")
        args[key] = value

    model = create_pretrained_model(args)

    # 清洗并重命名 checkpoint 中的参数 key
    raw_sd = { re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items() }
    new_sd = OrderedDict()
    for k, v in raw_sd.items():
        # 重命名 x_embedding
        m = re.match(r"pretrain_gnn\.representation_model\.embedding_dict\.x_embedding_(\d+)\.weight", k)
        if m:
            idx = int(m.group(1))
            nk = f"pretrain_gnn.representation_model.x_embedding{idx+1}.weight"
            new_sd[nk] = v
            continue
        # 重命名 edge_embedding
        m2 = re.match(
            r"pretrain_gnn\.representation_model\.gnns\.(\d+)\.embedding_dict\.edge_embedding_(\d+)\.weight",
            k
        )
        if m2:
            layer = int(m2.group(1))
            idx = int(m2.group(2))
            nk = f"pretrain_gnn.representation_model.gnns.{layer}.edge_embedding{idx+1}.weight"
            new_sd[nk] = v
            continue
        # 跳过不需要的 head 参数
        if "atom_task_lin" in k or "bond_task_lin2" in k:
            continue
        # 其余参数直接保留
        new_sd[k] = v

    model.load_state_dict(new_sd, strict=False)
    return model.to(device)


class PretrainedModel(nn.Module):
    def __init__(
        self,
        pretrain_gnn: nn.Module,
        contrastive_loss: nn.Module,
        mlm_loss: nn.Module,
    ):
        super(PretrainedModel, self).__init__()
        self.pretrain_gnn = pretrain_gnn
        self.contrastive_loss = contrastive_loss
        self.mlm_loss = mlm_loss
        
    @staticmethod
    def _get_positive_mask(weights):
        n = weights.shape[0]
        diag = torch.eye(n)
        l1 = torch.roll(diag, shifts=-n//2, dims=0)
        l2 = torch.roll(diag, shifts=n//2, dims=0)
        mask = l1 + l2 + diag
        return mask.bool()

    def forward(
        self,
        batch_tuple: tuple[Data, Data],
    ):
        data, masked_data = batch_tuple
        c_out = self.pretrain_gnn(data, masked=False)
        (
            c_out_masked,
            atom_output,
            bond_link_output,
            bond_class_output,
        ) = self.pretrain_gnn(masked_data, masked=True)
        maccs_fp = data.maccs.reshape(-1, 167)
        maccs_fp = torch.cat([maccs_fp, maccs_fp], dim=0)
        c_weights = 1. - F.cosine_similarity(
            maccs_fp.unsqueeze(1), maccs_fp.unsqueeze(0), dim=-1
        )
        c_weights[self._get_positive_mask(c_weights)] = 1.

        atoms_loss, bond_link_loss, bond_class_loss = self.mlm_loss(
            data, masked_data, atom_output, bond_link_output, bond_class_output
        )
        c_loss = self.contrastive_loss(c_out, c_out_masked, c_weights)

        return c_loss, atoms_loss, bond_link_loss, bond_class_loss