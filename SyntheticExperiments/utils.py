import random
import torch
import numpy as np
from torch_geometric.utils import from_networkx

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def create_data_list(graphs, label):
    # Function to convert the list of networkx graphs into a torch_geometric data list
    data_list = []
    for graph in graphs:
        data = from_networkx(graph)
        data.x = torch.full((data.num_nodes, 10), 0.1)
        data.y = label
        data.edge_weights = torch.ones(data.edge_index.size(1))
        data_list.append(data)
    return data_list

def create_masks(num_total_samples, train_ratio, val_ratio, test_ratio):
    # Function to create train, validation, and test masks

    # Calculate the number of samples for each split
    num_train = int(num_total_samples * train_ratio)
    num_val = int(num_total_samples * val_ratio)
    num_test = num_total_samples - num_train - num_val

    # Create a list of indices
    indices = list(range(num_total_samples))
    random.shuffle(indices)

    # Create the train, validation, and test masks
    train_mask = torch.tensor([i in indices[:num_train] for i in range(num_total_samples)], dtype=torch.bool)
    val_mask = torch.tensor([i in indices[num_train:num_train+num_val] for i in range(num_total_samples)], dtype=torch.bool)
    test_mask = torch.tensor([i in indices[num_train+num_val:] for i in range(num_total_samples)], dtype=torch.bool)

    return train_mask, val_mask, test_mask

def create_mask_without_test(num_total_samples, train_ratio, val_ratio):
    # Function to create train and validation
    # Calculate the number of samples for each split
    assert train_ratio+val_ratio == 1
    num_train = int(num_total_samples * train_ratio)

    # Create a list of indices
    indices = list(range(num_total_samples))
    random.shuffle(indices)

    # Create the train and validation masks
    train_mask = torch.tensor([i in indices[:num_train] for i in range(num_total_samples)], dtype=torch.bool)
    val_mask = torch.tensor([i in indices[num_train:] for i in range(num_total_samples)], dtype=torch.bool)

    return train_mask, val_mask


def evaluate_specific(out, labels, label):
    """
    Calculates the accuracy between the prediction and the ground truth for a specific label.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :param label: the label we want to specificly evaluate
    :returns: int accuracy
    """
    preds = out.argmax(dim=1)
    correct = preds == labels
    correct = correct[labels==label]
    acc = int(correct.sum()) / int(correct.size(0))
    return acc