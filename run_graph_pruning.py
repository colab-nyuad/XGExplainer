import argparse
import torch

from SyntheticExperiments.graph_motifs_plus import *

parser = argparse.ArgumentParser(description='Run Graph Pruning Experiment')

'''
Experiment args
'''
parser.add_argument('--train_model', default=False, action='store_true', 
                    help='Whether to train the GNNs')
parser.add_argument('--evaluate_GNN', default=False, action='store_true', 
                    help='Whether to evaluate the GNNs') 
parser.add_argument('--compute_lambda', default=False, action='store_true', help='Whether to compute the lambda value of the validation set')
'''
Dataset args
'''
parser.add_argument('--train_ratio', type=float, default=0.8) 
parser.add_argument('--valid_ratio', type=float, default=0.1) 
parser.add_argument('--test_ratio', type=float, default=0.1)
parser.add_argument('--num_samples', type=int, default=200, help="The number of samples for each data type")
parser.add_argument('--num_nodes_base', type=int, default=20)

"""
Training args
"""
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--b', type=float, default=0.1, help="probability of randomized edge dropping")
parser.add_argument('--_eval', default=False, action='store_true', 
                    help='Whether to train/evaluate GNN_eval') 
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip_max', type=float, default=2.0)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--early_stopping', type=int, default=1000)
parser.add_argument('--save_dir', type=str, default="default", help="directory to store, use for hyperparam tuning")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args, flush=True)
    if args.train_model:
        train_models(args)
    if args.evaluate_GNN:
        evaluate_gnns(args)