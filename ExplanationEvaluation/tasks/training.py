import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.tasks.eval_xg import Randomized_Edge_Dropping

from IPython.display import clear_output
@torch.no_grad()
def live_plot(data_dict, figsize=(7,5), title=''):
    clear_output(wait=True)
    for label,data in data_dict.items():
        plt.plot(data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='upper right') # the plot evolves to the right
    plt.show()

def create_data_list(graphs, features, labels, mask):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    indices = np.argwhere(mask).squeeze()
    data_list = []
    for i in indices:
        x = torch.tensor(features[i])
        edge_index = torch.tensor(graphs[i])
        y = torch.tensor(labels[i].argmax())
        edge_weights = torch.ones(edge_index.size(1))
        data = Data(x=x, edge_index=edge_index, y=y, edge_weights = edge_weights)
        data_list.append(data)
    return data_list


def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    preds = out.argmax(dim=1)
    correct = preds == labels
    acc = int(correct.sum()) / int(correct.size(0))
    return acc



def store_checkpoint(paper, dataset, model, train_acc, val_acc, test_acc, epoch=-1, _eval=False):
    """
    Store the model weights at a predifined location.
    :param paper: str, the paper 
    :param dataset: str, the dataset
    :param model: the model who's parameters we whish to save
    :param train_acc: training accuracy obtained by the model
    :param val_acc: validation accuracy obtained by the model
    :param test_acc: test accuracy obtained by the model
    :param epoch: the current epoch of the training process
    :param _eval: whether the trained model is an GNN_EVAL
    :retunrs: None
    """

    if _eval:
        save_dir = f"./checkpoints/{paper}/{dataset}/GNN_EVAL"
    else:
        save_dir = f"./checkpoints/{paper}/{dataset}"

    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))


def load_best_model(best_epoch, paper, dataset, model, eval_enabled, _eval=False):
    """
    Load the model parameters from a checkpoint into a model
    :param best_epoch: the epoch which obtained the best result. use -1 to chose the "best model"
    :param paper: str, the paper 
    :param dataset: str, the dataset
    :param model: the model who's parameters overide
    :param eval_enabled: wheater to activate evaluation mode on the model or not
    :param _eval: whether the trained model is an GNN_EVAL
    :return: model with pramaters taken from the checkpoint
    """
    if _eval:
        save_dir = f"./checkpoints/{paper}/{dataset}/GNN_EVAL"
    else:
        save_dir = f"./checkpoints/{paper}/{dataset}"

    if best_epoch == -1:
        checkpoint = torch.load(f"{save_dir}/best_model")
    else:
        checkpoint = torch.load(f"{save_dir}/model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_enabled: model.eval()

    return model


def train_node(_dataset, _paper, args, _eval=False, device="cpu"):
    """
    Train a explainer to explain node classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    :param _eval: True or False on whether the trained model is GNN_EVAL or not
    :param device: cuda or cpu
    """
    graph, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    x = torch.tensor(features) ##NODE FEATURES
    edge_index = torch.tensor(graph)
    labels = torch.tensor(labels)
    edge_weights = torch.ones(edge_index.size(1))

    data = Data(x=x, edge_index=edge_index, edge_weights=edge_weights, y=labels, train_mask = train_mask, val_mask=val_mask, test_mask=test_mask)
    data.to(device)

    model = model_selector(_paper, _dataset, False)
    model.to(device)

    # Define graph
    if _eval:
        print('Training an GNN_EVAL')
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()


    best_val_acc = 0.0
    best_epoch = 0

    losses = {'train':[], 'val': []}

    for epoch in range(0, args.epochs):
        model.train()
        optimizer.zero_grad()

        if _eval:  ##PERFORM RANDOM EDGE DROPPING/MASKING HERE
            sample_adj = Randomized_Edge_Dropping(data.edge_index, args.b, device)
            sample_edge_weights = torch.ones(sample_adj.size(1)).to(device)
            out = model(data.x, sample_adj, sample_edge_weights)
        else:
            out = model(data.x, data.edge_index, data.edge_weights)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        optimizer.step()

        if args.eval_enabled: model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_weights)

        # Evaluate train
        train_acc = evaluate(out[data.train_mask], data.y[data.train_mask])
        test_acc = evaluate(out[data.test_mask], data.y[data.test_mask])
        val_acc = evaluate(out[data.val_mask], data.y[data.val_mask])
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])

        losses['train'].append(loss.cpu())
        losses['val'].append(val_loss.cpu())

        live_plot(losses)

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc: # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, best_epoch, _eval)

        if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
            break

    model = load_best_model(best_epoch, _paper, _dataset, model, args.eval_enabled, _eval)
    out = model(data.x, data.edge_index, data.edge_weights)

    # Train eval
    train_acc = evaluate(out[data.train_mask], data.y[data.train_mask])
    test_acc = evaluate(out[data.test_mask], data.y[data.test_mask])
    val_acc = evaluate(out[data.val_mask], data.y[data.val_mask])

    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, -1, _eval)


def train_graph(_dataset, _paper, args, _eval=False, device="cpu"):
    """
    Train a explainer to explain graph classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    :param _eval: True or False on whether the trained model is GNN_EVAL or not
    :param device: cuda or cpu
    """
    graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    train_set = create_data_list(graphs, features, labels, train_mask)
    val_set = create_data_list(graphs, features, labels, val_mask)
    test_set = create_data_list(graphs, features, labels, test_mask)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    model = model_selector(_paper, _dataset, False)
    model.to(device)

    if _eval:
        print('Training an GNN_EVAL')

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    losses = {'train':[], 'val': []}

    for epoch in range(0, args.epochs):
        model.train()

        # Use pytorch-geometric batching method
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            if _eval:  ##PERFORM RANDOM EDGE DROPPING/MASKING HERE
                sample_adj = Randomized_Edge_Dropping(data.edge_index, args.b, device=device)
                sample_edge_weights = torch.ones(sample_adj.size(1)).to(device)
                out = model(data.x, sample_adj, data.batch, sample_edge_weights)
            else:
                out = model(data.x, data.edge_index, data.batch, data.edge_weights)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
            optimizer.step()

        model.eval()
        # Evaluate train
        with torch.no_grad():
            train_sum = 0
            loss = 0
            for data in train_loader:
                data.to(device)
                out = model(data.x, data.edge_index, data.batch, data.edge_weights)
                loss += criterion(out, data.y)
                preds = out.argmax(dim=1)
                train_sum += (preds == data.y).sum()
            train_acc = int(train_sum) / int(len(train_set))
            train_loss = float(loss) / int(len(train_loader))

            eval_data = next(iter(test_loader)) # Loads all test samples
            eval_data.to(device)
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch, eval_data.edge_weights)
            test_acc = evaluate(out, eval_data.y)

            eval_data = next(iter(val_loader)) # Loads all eval samples
            eval_data.to(device)
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch, eval_data.edge_weights)
            val_acc = evaluate(out, eval_data.y)
            val_loss = criterion(out, eval_data.y)
        
        losses['train'].append(train_loss)
        losses['val'].append(val_loss.cpu())
        live_plot(losses)

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, best_epoch, _eval)

        # Early stopping
        if epoch - best_epoch > args.early_stopping:
            break

    model = load_best_model(best_epoch, _paper, _dataset, model, args.eval_enabled, _eval)

    with torch.no_grad():
        train_sum = 0
        for data in train_loader:
            data.to(device)
            out = model(data.x, data.edge_index, data.batch, data.edge_weights)
            preds = out.argmax(dim=1)
            train_sum += (preds == data.y).sum()
        train_acc = int(train_sum) / int(len(train_set))

        eval_data = next(iter(test_loader))
        eval_data.to(device)
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch, eval_data.edge_weights)
        test_acc = evaluate(out, eval_data.y)

        eval_data = next(iter(val_loader))
        eval_data.to(device)
        out = model(eval_data.x, eval_data.edge_index, eval_data.batch, eval_data.edge_weights)
        val_acc = evaluate(out, eval_data.y)

    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, -1, _eval)