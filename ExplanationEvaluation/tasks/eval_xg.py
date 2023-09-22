import torch
from torch_geometric.utils import *
import pandas as pd
import os

@torch.no_grad()
def Randomized_Edge_Dropping(E, b=0.5, device="cpu"):
    '''
    Create a dataset for samples
    E: Edge Index
    b: the bernoulli prob. of masking for adjacency matrix (random edge dropping probability)
    device: cpu or cuda
    '''
    edge_drop_prob = torch.rand((E.shape[1],)).to(device)
    mask = edge_drop_prob.ge(b)
    sample_edge_index = E.T[mask==True].T

    return sample_edge_index

def write_results(auc, auc_std, config, _eval):
    '''
    Write the result of explanation
    '''

    save_dir = 'results'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    if _eval:
        file = os.path.join(save_dir, f"XGExplainer_{config.explainer}_{config.dataset}.csv")
    else:
        file = os.path.join(save_dir, f"{config.explainer}_{config.dataset}.csv")

    results = {"dataset": config.dataset,
        'Explanation AUC Avg': [auc],
        'Explanation AUC Std': auc_std,
        }

    results = pd.DataFrame(results)
    file_exists = os.path.exists(file)

    if file_exists:
        results.to_csv(file, mode='a', header=False, index=False)
    else:
        results.to_csv(file, mode='w', index=False)