import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):

    def __init__(self, temperature=0.8):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, input, weights=None):
        input = F.normalize(input, dim=-1)
        logits = input @ input.T
        logits = logits * self.temperature  # scale via temperature
        exp_logits = torch.exp(logits)

        if weights is not None:
            weights = self.weights_normalize(weights)
            exp_logits = exp_logits * weights

        logits_sum = torch.sum(exp_logits, dim=1)
        prob = torch.log(1.0e-5 + (exp_logits / logits_sum[:, None]))
        positive_prob = torch.diag(prob)
        loss = -torch.mean(positive_prob, dim=0)

        return loss

    @staticmethod
    def weights_normalize(w: torch.Tensor):
        w.fill_diagonal_(0.0)
        epsilon = 1e-8
        w = w / (w.sum(dim=1, keepdim=True) + epsilon)
        w.fill_diagonal_(1.0)
        w = torch.clamp(w, 0.01, 0.99)
        return w


class MLMLoss(nn.Module):

    def __init__(self):
        super(MLMLoss, self).__init__()
        self.atom_loss = nn.CrossEntropyLoss()
        self.bond_link_loss = nn.CrossEntropyLoss()
        self.bond_class_loss = nn.CrossEntropyLoss()

    def forward(self, data, masked_data, atom_output, bond_link_output, bond_class_output):
        x = data.x
        edge_attr = data.edge_attr
        x_masked = masked_data.x
        edge_attr_masked = masked_data.edge_attr
        edge_add_seg = masked_data.edge_add_seg
        # Create a boolean mask where each element is True if x[i] == [9, 4, 5]
        atom_mask = (x_masked == torch.tensor([9, 4, 5], dtype=torch.long).to(
            x.device)).all(dim=-1)

        # Filter the output and x_masked using the boolean mask
        mlm_atom_output = atom_output[atom_mask]
        mlm_atom_label = x[atom_mask]

        # Compute the label indices from mlm_label
        mlm_atom_label = (mlm_atom_label[:, 0] * 20 + mlm_atom_label[:, 1] * 5 +
                     mlm_atom_label[:, 2]).type(torch.long)

        atoms_loss = self.atom_loss(mlm_atom_output, mlm_atom_label)
        
        edge_mask = (edge_attr_masked == torch.tensor([4, 3], dtype=torch.long).to(
            edge_attr.device)).all(dim=-1)
        
        edge_link_output = bond_link_output[edge_mask]
        edge_link_label = edge_add_seg[edge_mask]
        
        bond_link_loss = self.bond_link_loss(edge_link_output, edge_link_label.to(torch.long))
        
        mlm_bond_output = bond_class_output[edge_mask & edge_add_seg]
        mlm_bond_label = edge_attr[edge_mask[edge_add_seg]]
        
        mlm_bond_label = (mlm_bond_label[:, 0] * 3 + mlm_bond_label[:, 1]).type(torch.long)
        
        bond_class_loss = self.bond_class_loss(mlm_bond_output, mlm_bond_label)
        
        return atoms_loss, bond_link_loss, bond_class_loss
        
        
class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = torch.eye(2 * batch_size)
        l1 = torch.roll(diag, shifts=-batch_size, dims=0)
        l2 = torch.roll(diag, shifts=batch_size, dims=0)
        mask = diag + l1 + l2
        return (1 - mask).type(torch.bool)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (2N, 1, C)
        # y shape: (1, C, 2N)
        return v
    
    @staticmethod
    def _cosine_simililarity(x, y):
        # x shape: (2N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (2N, 2N)
        v = torch.nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, weights=None):
        batch_size = zis.shape[0]
        # 2N, C
        representations = torch.cat([zjs, zis], dim=0)

        # 2N, 2N
        logits = self.similarity_function(representations, representations)
        logits /= self.temperature
        
        if weights is not None:
            logits = logits * weights
            
        exp_logits = torch.exp(logits)
        '''
        if weights is not None:
            exp_logits = exp_logits * weights
        '''
        # filter out the scores from the positive samples
        l_pos = torch.diag(exp_logits, batch_size)
        r_pos = torch.diag(exp_logits, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool).to(representations.device)
        negatives = exp_logits[mask_samples_from_same_repr].view(2 * batch_size, -1)

        exp_logits = torch.cat((positives, negatives), dim=1)

        logits_sum = torch.sum(exp_logits, dim=1)
        prob = -torch.log(1.0e-5 + positives / logits_sum)  
        loss = torch.mean(prob)

        return loss