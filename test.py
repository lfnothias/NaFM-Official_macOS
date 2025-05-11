import argparse

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, recall_score,
                             accuracy_score, precision_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from tqdm import tqdm

from gnn.datasets import *
from gnn.tune_module import LNNP as FinetunedLNNP


def get_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument(
        "--dataset-arg",
        default='Class',
        type=str,
        help="Dataset argument",
    )

    parser.add_argument(
        "--dataset",
        default='Ontology',
        type=str,
        help="Finetuned Dataset name",
        choices=['Lotus', 'Ontology', 'Regression', 'External', 'BGC'],
    )

    parser.add_argument(
        "--dataset-root",
        default='./downstream_data/Ontology',
        type=str,
        help="Data storage directory",
    )

    parser.add_argument(
        "--log-dir",
        default='logs-finetune-tune',
        type=str,
        help="splits storage directory",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="checkpoint storage directory",
    )

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.dataset == 'Lotus':
        dataset = Lotus(root=args.dataset_root)
        
    elif args.dataset == 'Ontology':
        dataset = Ontology(root=args.dataset_root,
                           dataset_arg=args.dataset_arg)
    elif args.dataset == 'External':
        dataset = External(root=args.dataset_root, dataset_arg=args.dataset_arg)
        
    elif args.dataset == 'Regression':
        dataset = NPC(root=args.dataset_root, dataset_arg=args.dataset_arg)
        
    elif args.dataset == 'BGC':
        dataset = BGC(root=args.dataset_root)
    
    else:
        raise ValueError('Dataset not found')

    splits = np.load(f'{args.log_dir}/splits.npz')
    checkpoint_path = f'{args.log_dir}/{args.checkpoint}'
    
    test_set = Subset(dataset, splits.f.idx_test)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    if args.dataset != 'Regression':
        model = FinetunedLNNP.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            num_classes=dataset.num_class)
        n_classes = dataset.num_class
    else:
        model = FinetunedLNNP.load_from_checkpoint(
            checkpoint_path=checkpoint_path, num_classes=1)
        n_classes = 1
        
    model = model.to(device)
    model.eval()
    
    # print number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    all_outputs = []
    all_labels = []

    for inputs in tqdm(test_loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            if args.dataset != 'Regression':
                outputs = torch.softmax(outputs, dim=-1)
            all_outputs.append(outputs.detach().cpu().numpy())
            all_labels.append(inputs.label.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)  # (n_samples, n_classes)
    all_labels = np.concatenate(all_labels)  # (n_samples, )
    if args.dataset == 'External': # External dataset has 2 classes
        auprc = average_precision_score(all_labels, all_outputs[:, 1])
        recall = recall_score(all_labels, np.argmax(all_outputs, axis=1))
        accuracy = accuracy_score(all_labels, np.argmax(all_outputs, axis=1))
        precision = precision_score(all_labels, np.argmax(all_outputs, axis=1))
        print(f'AUPRC: {auprc}')
        print(f'Recall: {recall}')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')

    elif args.dataset == 'BGC':
        roc_auc_scores = []
        valid_auprc_scores = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                roc_auc_scores.append(roc_auc_score(all_labels[:, i], all_outputs[:, i]))
                valid_auprc_scores.append(average_precision_score(all_labels[:, i], all_outputs[:, i]))
            else:
                roc_auc_scores.append(float('nan'))
                valid_auprc_scores.append(float('nan'))
        roc_auc = np.nanmean(roc_auc_scores)
        auprc = np.nanmean(valid_auprc_scores)
        accuracy = accuracy_score(all_labels, np.round(all_outputs)) 
        recall = recall_score(all_labels, np.round(all_outputs), average='macro')

        print(f'AUPRC: {auprc:.6f}')
        print(f'ROC AUC: {roc_auc:.6f}')
        print(f'Accuracy: {accuracy:.6f}')
        print(f'Recall: {recall:.6f}')

    elif args.dataset != 'Regression':
        binarized_labels = label_binarize(
            all_labels, classes=[i for i in range(n_classes)])
        auprc, count = np.zeros(n_classes), 0
        for i in range(n_classes):
            # Only compute AUC for valid classes
            if len(np.unique(binarized_labels[:, i])) > 1:
                auprc[i] = average_precision_score(binarized_labels[:, i],
                                                   all_outputs[:, i])
                count += 1
        auprc = np.sum(auprc) / count
        recall = recall_score(all_labels,
                              np.argmax(all_outputs, axis=1),
                              average='macro')
        accuracy = accuracy_score(all_labels, np.argmax(all_outputs, axis=1))
        print(f'AUPRC: {auprc}')
        print(f'Recall: {recall}')
        print(f'Accuracy: {accuracy}')

    else:
        print(f'RMSE: {np.sqrt(np.mean((all_labels - all_outputs)**2))}')
        print(f'MAE: {np.mean(np.abs(all_labels - all_outputs))}')


if __name__ == "__main__":
    main()