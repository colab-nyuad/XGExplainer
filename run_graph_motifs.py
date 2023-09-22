import argparse
import torch

from SyntheticExperiments.graph_motifs import *

parser = argparse.ArgumentParser(description='Run the experiment of Generalizing Abstruct Structures')
"""
Experiment args
"""
parser.add_argument('--train_model', default=False, action='store_true', 
                    help='Whether to train the GNNs')
parser.add_argument('--evaluate_GNN', default=False, action='store_true', 
                    help='Whether to evaluate the GNNs') 
parser.add_argument('--_eval', default=False, action='store_true', 
                    help='Whether to train/evaluate GNN_eval') 
parser.add_argument('--save_dir', type=str, default="default", 
                    help="directory to store, use for parallel training with different hyperparams")
'''
Dataset args
'''
parser.add_argument('--train_ratio', type=float, default=0.8) 
parser.add_argument('--valid_ratio', type=float, default=0.2)
parser.add_argument('--num_nodes_base', type=int, default=20)
parser.add_argument('--num_samples', type=int, default = 100, help="number of samples per motif on the train+val set")
parser.add_argument('--num_samples_eval', type=int, default = 100, help="number of samples per motif on the test set")
parser.add_argument('--min', type=int, default=11, help="minimum number of nodes in iid set")
parser.add_argument('--max', type=int, default=20, help="maximum number of nodes in iid set")
parser.add_argument('--ood_min', type=int, default=6, help="minimum number of nodes in ood set")
parser.add_argument('--ood_max', type=int, default=10, help="maximum number of nodes in ood set")

"""
Training args
"""
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--b', type=float, default=0.1, help="probability of randomized edge dropping")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip_max', type=float, default=2.0)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--early_stopping', type=int, default=300)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args, flush=True)
    if args.train_model:
        train_models(args)
    if args.evaluate_GNN:
        evaluate_gnns(args)