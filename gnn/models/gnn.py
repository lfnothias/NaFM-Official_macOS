import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.pool import global_mean_pool
from .layers import GATConv, GraphSAGEConv, GINConv, GCNConv, Linear_Block

num_atom_type = 9
num_chirality_tag = 4
num_charge = 5

num_bond_type = 5
num_bond_direction = 4


class GNN(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, num_layer, emb_dim, drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type + 1, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag + 1, emb_dim)
        self.x_embedding3 = torch.nn.Embedding(num_charge + 1, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # List of norms
        self.norms = torch.nn.ModuleList()
        for _ in range(num_layer):
            # self.norms.append(GraphNorm(emb_dim))
            self.norms.append(nn.LayerNorm(emb_dim))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        h = (self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1]) +
             self.x_embedding3(x[:, 2]))

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            # h = self.norms[layer](h, batch)
            h = self.norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)

        return h


class PretrainedGNN(nn.Module):

    def __init__(
        self,
        num_layer=8,
        emb_dim=512,
        drop_ratio=0,
        gnn_type="gin",
        feat_dim=1024,
    ):
        super(PretrainedGNN, self).__init__()

        self.representation_model = GNN(num_layer, emb_dim, drop_ratio,
                                        gnn_type)

        self.MLM = nn.Sequential(
            nn.Linear(emb_dim, feat_dim),
            nn.LeakyReLU(),
        )
        self.atom_task_lin = nn.Linear(
            feat_dim, num_atom_type * num_chirality_tag * num_charge)
        self.bond_task_lin = nn.Linear(2 * feat_dim, 2)
        self.bond_task_lin2 = nn.Linear(2 * feat_dim, (num_bond_type - 1) *
                                        (num_bond_direction - 1))

        self.contrast = nn.Sequential(
            nn.Linear(emb_dim, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, feat_dim // 2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.contrast[0].weight.data)
        nn.init.normal_(self.contrast[0].bias.data, mean=0, std=4.0e-4)
        nn.init.zeros_(self.contrast[-1].weight.data)
        nn.init.normal_(self.contrast[-1].bias.data, mean=0, std=4.0e-4)

    def forward(self, data, masked=False):
        h = self.representation_model(data)

        c = global_mean_pool(h, data.batch)
        c = self.contrast(c)
        if masked:
            atom_embedding = self.MLM(h)
            atom_output = self.atom_task_lin(atom_embedding)
            bond_embedding = torch.cat(
                [
                    atom_embedding[data.edge_index[0]],
                    atom_embedding[data.edge_index[1]],
                ],
                dim=-1,
            )
            bond_link_output = self.bond_task_lin(bond_embedding)
            bond_class_output = self.bond_task_lin2(bond_embedding)
            return c, atom_output, bond_link_output, bond_class_output
        else:
            return c


class FinetunedGNN(nn.Module):

    def __init__(
        self,
        num_layer=8,
        emb_dim=512,
        drop_ratio=0,
        linear_drop_ratio=0.1,
        gnn_type="gin",
        num_classes=1,
        include_target=False,
    ):
        super(FinetunedGNN, self).__init__()

        self.representation_model = GNN(num_layer, emb_dim, drop_ratio,
                                        gnn_type)

        self.output_block = Linear_Block(emb_dim, emb_dim // 2, num_classes,
                                         linear_drop_ratio)
        
        if include_target:
            self.target_emb = nn.Embedding(1971, emb_dim)
            self.target_combine = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, data):
        representation = self.representation_model(data)

        graph_embedding = global_mean_pool(representation, data.batch)

        if hasattr(self, "target_emb"):
            target_embedding = self.target_emb(data.target_id)
            graph_embedding = self.target_combine(torch.cat([graph_embedding, target_embedding], dim=-1))

        output = self.output_block(graph_embedding)

        return output