import time
import json
import os
import copy 

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.datasets.ground_truth_loaders import load_dataset_ground_truth
from ExplanationEvaluation.evaluation.AUCEvaluation import AUCEvaluation
from ExplanationEvaluation.evaluation.EfficiencyEvaluation import EfficiencyEvluation
from ExplanationEvaluation.explainers.GNNExplainer import GNNExplainer
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.utils.plotting import plot
from ExplanationEvaluation.tasks.eval_xg import write_results

def get_classification_task(graphs):
    """
    Given the original data, determines if the task as hand is a node or graph classification task
    :return: str either 'graph' or 'node'
    """
    if isinstance(graphs, list): # We're working with a model for graph classification
        return "graph"
    else:
        return "node"

def to_torch_graph(graphs, task):
    """
    Transforms the numpy graphs to torch tensors depending on the task of the model that we want to explain
    :param graphs: list of single numpy graph
    :param task: either 'node' or 'graph'
    :return: torch tensor
    """
    if task == 'graph':
        return [torch.tensor(g) for g in graphs]
    else:
        return torch.tensor(graphs)


def select_explainer(explainer, model, graphs, features, task, epochs, lr, reg_coefs, temp=None, sample_bias=None, hops = 3):
    """
    Select the explainer we which to use.
    :param explainer: str, "PG"
    :param model: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :params sample_bias: the bias we add when sampling random graphs. 
    :param hops: the number of layers in the GNN
    """
    if explainer == "PG":
        return PGExplainer(model, graphs, features, task, epochs=epochs, lr=lr, reg_coefs=reg_coefs, temp=temp, sample_bias=sample_bias, hops = hops)
    elif explainer == "GNN":
        return GNNExplainer(model, graphs, features, task, epochs=epochs, lr=lr, reg_coefs=reg_coefs, hops=hops)
    else:
        raise NotImplementedError("Unknown explainer type")

def run_experiment(task, inference_eval, auc_evaluation, explainer, indices):
    """
    Runs an experiment.
    We generate explanations for given indices and calculate the AUC score.
    :param task: "node" or "graph"
    :param inference_eval: object for measure the inference speed
    :param auc_evaluation: a metric object, which calculate the AUC score
    :param explainer: the explainer we wish to obtain predictions from
    :param indices: indices over which to evaluate the auc
    :returns: AUC score, inference speed
    """
    inference_eval.start_prepate()
    explainer.prepare(indices)

    inference_eval.start_explaining()
    explanations = []

    for idx in tqdm(indices):
        graph, expl = explainer.explain(idx)
        explanations.append((graph, expl))

    inference_eval.done_explaining()     

    auc_score = auc_evaluation.get_score(explanations)
    time_score = inference_eval.get_score(explanations)

    return auc_score, time_score


def run_qualitative_experiment(explainer, indices, labels, config, explanation_labels):
    """
    Plot the explaination generated by the explainer
    :param explainer: the explainer object
    :param indices: indices on which we validate
    :param labels: predictions of the explainer
    :param config: dict holding which subgraphs to plot
    :param explanation_labels: the ground truth labels 
    """
    for idx in indices:
        graph, expl = explainer.explain(idx)
        plot(graph, expl, labels, idx, config.thres_min, config.thres_snip, config.dataset, config, explanation_labels)

def replication(config, _eval=True, run_qual=True):
    """
    Perform the replication study.
    First load a pre-trained model.
    Then we train our expainer.
    Followed by obtaining the generated explanations.
    And saving the obtained AUC score in a json file.
    :param config: a dict containing the config file values
    :param _eval: True if using GNN_eval as evaluator, False if using GNN as evaluator.
    :param run_qual: whether to run qualitative experiment
    """
    # Load complete dataset
    graphs, features, labels, _, _, test_mask = load_dataset(config.dataset)
    task = get_classification_task(graphs)

    features = torch.tensor(features)
    labels = torch.tensor(labels)
    graphs = to_torch_graph(graphs, task) #for graph dataset, graphs are list of edge index, features 

    if _eval:
        print("Loading GNN_eval as evaluator")
    else:
        print("Loading GNN as evaluator")
        
    model, checkpoint = model_selector(config.model,
                                        config.dataset,
                                        pretrained=True,
                                        return_checkpoint=True,
                                        _eval=_eval)                  


    # Get ground_truth for every node
    explanation_labels, indices = load_dataset_ground_truth(config.dataset)

    # Get explainer
    explainer = select_explainer(config.explainer,
                                 model=model,
                                 graphs=graphs,
                                 features=features,
                                 task=task,
                                 epochs=config.epochs,
                                 lr=config.lr,
                                 reg_coefs=[config.reg_size,
                                            config.reg_ent],
                                 temp=config.temps,
                                 sample_bias=config.sample_bias)

    # Get evaluation methods
    auc_evaluation = AUCEvaluation(task, explanation_labels, indices)
    inference_eval = EfficiencyEvluation()

    # Perform the evaluation 10 times
    auc_scores = []
    times = []

    for idx, s in enumerate(config.seeds):
        print(f"Run {idx} with seed {s}")
        # Set all seeds needed for reproducibility
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        np.random.seed(s)

        inference_eval.reset()
        
        auc_score, time_score = run_experiment(task, inference_eval, auc_evaluation, explainer, indices)

        if idx == 0 and run_qual: # We only run the qualitative experiment once
            run_qualitative_experiment(explainer, indices, labels, config, explanation_labels)

        auc_scores.append(auc_score)
        print("score:",auc_score)
        times.append(time_score)
        print("time_elased:",time_score)

    auc = np.mean(auc_scores)
    auc_std = np.std(auc_scores)
    inf_time = np.mean(times) / 10

    print("writing the results...")
    write_results(auc, auc_std, config, _eval)
        
    return (auc, auc_std), inf_time