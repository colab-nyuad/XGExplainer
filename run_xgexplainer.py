import argparse
import torch
import numpy as np
import os

from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication import replication

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")

def main(args):
    # Get XGExplainer Config
    
    config_path = f"./ExplanationEvaluation/configs/replication/explainers/EVAL/{args.explainer}/{args.dataset}.json"
    config = Selector(config_path).args.explainer

    if args.all_seeds == False: 
        config.seeds = [0] #to only run the evaluation once
    (auc, auc_std), inf_time = replication(config, run_qual=args.run_qual)
    print((auc, auc_std), inf_time)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bashapes') # bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
    parser.add_argument('--all_seeds', default=False, action='store_true') #whether to evaluate on all seeds
    parser.add_argument('--run_qual',default=False, action='store_true') #whether to run qualitative experiment
    parser.add_argument('--explainer', type=str, default='pgexplainer', choices=['pgexplainer', 'gnnexplainer']) # the base explainer for XGExplainer, either PGExplainer or GNNExplainer
    args = parser.parse_args()
    print(args, flush=True)
    main(args)