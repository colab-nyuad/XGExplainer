from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.training import train_node, train_graph
import torch
import numpy as np
import argparse
import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")

parser = argparse.ArgumentParser(description='Train GNN eval on Node and Graph Classification Datasets')

'''
Experiment Args
'''
parser.add_argument('--dataset', default="bashapes", choices=["bashapes", "bacommunity", "treecycles", "treegrids", "ba2motifs", "mutag"])

def main(args):
    print(f"Working on {args.dataset} dataset")
    
    # The model architecture is based on the GNNExplainer Paper
    _model = 'gnn'
    print(f"The model will be the one based on {_model} paper")

    #Taking the GNN_eval configuration
    eval_config_path = f"./ExplanationEvaluation/configs/replication/eval/model_{_model}_{args.dataset}.json"
    eval_config = Selector(eval_config_path).args

    print("The probability of edge_dropping for GNN_EVAL is ", eval_config.model.b)
    print("GNN_EVAL will have #epochs ", eval_config.model.epochs)

    # setting the seed
    torch.manual_seed(eval_config.model.seed)
    torch.cuda.manual_seed(eval_config.model.seed)
    np.random.seed(eval_config.model.seed)

    #getting the config representation of the dataset and paper
    _dataset = eval_config.model.dataset
    _paper = eval_config.model.paper

    _eval = True #indicates that this is GNN_EVAL, not the full model GNN
    ##TRAINING
    if _dataset[:3] == "syn":
        train_node(_dataset, _paper, eval_config.model, _eval, device)
    elif _dataset == "ba2" or _dataset == "mutag":
        train_graph(_dataset, _paper, eval_config.model, _eval, device)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args, flush=True)
    main(args)