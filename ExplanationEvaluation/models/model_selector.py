import torch
import os

from ExplanationEvaluation.models.GNN_paper import NodeGCN as GNN_NodeGCN
from ExplanationEvaluation.models.GNN_paper import GraphGCN as GNN_GraphGCN
from ExplanationEvaluation.models.GNN_paper import GraphGIN as GNN_GraphGIN

def string_to_model(paper, dataset):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: torch.nn.module models
    """
    if paper == "GNN":
        if dataset in ['syn1']:
            return GNN_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return GNN_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return GNN_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return GNN_NodeGCN(10, 2)
        elif dataset == "ba2":
            return GNN_GraphGCN(10, 2)
        elif dataset == "mutag":
            return GNN_GraphGCN(14, 2)
        elif "Graph_Motifs_Plus" in dataset:
            return GNN_GraphGIN(10, 7)
        elif "Graph_Motifs" in dataset:
            return GNN_GraphGIN(10, 7)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_pretrained_path(paper, dataset, _eval=False):
    """
    Given a paper and dataset loads the pre-trained model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param _eval: whether to return pretrained GNN_EVAL or not
    :returns: str; the path to the pre-trined model parameters.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if _eval:
        path = f"{dir_path}/pretrained/{paper}/{dataset}/GNN_EVAL/best_model"
    else:
        path = f"{dir_path}/pretrained/{paper}/{dataset}/best_model"
    return path


def model_selector(paper, dataset, pretrained=True, return_checkpoint=False, _eval=False, device='cpu'):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :param _eval: whether to return EVAL-XG or the original model
    :param device: cpu or cuda
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset)
    if pretrained:
        path = get_pretrained_path(paper, dataset, _eval)
        checkpoint = torch.load(path, map_location=torch.device(device))
        ## EDITED
        try:
            renamed_state_dict = {}
            for key in checkpoint['model_state_dict']:
                if key.startswith('conv') and key.endswith('weight'):
                    new_key = key[:5] + '.lin' + key[5:]
                    renamed_state_dict[new_key] = (checkpoint['model_state_dict'][key]).T
                else:
                    renamed_state_dict[key] = checkpoint['model_state_dict'][key]
            model.load_state_dict(renamed_state_dict)
        ## EDITED
        except:
            model.load_state_dict(checkpoint['model_state_dict'])
        #model.load_state_dict(checkpoint['model_state_dict']) COMMENTED OUT
        print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model