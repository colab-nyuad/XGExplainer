import random
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import os
import csv
from ExplanationEvaluation.tasks.eval_xg import Randomized_Edge_Dropping
from ExplanationEvaluation.models.GNN_paper import GraphGIN as GNN_GraphGIN
from ExplanationEvaluation.models.model_selector import model_selector
from ExplanationEvaluation.tasks.training import *

from SyntheticExperiments.create_datasets import *
from SyntheticExperiments.utils import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")

def train_models(args):
    if args.save_dir == "default":
        dataset_name = "Graph_Motifs_Plus"
    if args.save_dir != "default":
        dataset_name = f"Graph_Motifs_Plus_{args.save_dir}"

    num_nodes_motif = 6

    circular_ladder = [create_motifs_with_varying_unimp("circular_ladder", 1.0, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
    complete = [create_motifs_with_varying_unimp("complete", 1.0, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
    grid = [create_motifs_with_varying_unimp("grid", 1.0, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
    lollipop = [create_motifs_with_varying_unimp("lollipop", 1.0, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
    wheel = [create_motifs_with_varying_unimp("wheel", 1.0, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
    star = [create_motifs_with_varying_unimp("star", 1.0, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
    cycle = [create_motifs_with_varying_unimp("cycle", 1.0, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]

    num_classes = 7

    circular_ladder_data_list = create_data_list(circular_ladder, 0)
    complete_data_list = create_data_list(complete, 1)
    grid_data_list = create_data_list(grid, 2)
    lollipop_data_list = create_data_list(lollipop, 3)
    wheel_data_list = create_data_list(wheel, 4)
    star_data_list = create_data_list(star, 5)
    cycle_data_list = create_data_list(cycle, 6)

    # Combine the data lists into a single dataset
    dataset = circular_ladder_data_list+complete_data_list+grid_data_list+lollipop_data_list\
                +wheel_data_list+star_data_list+cycle_data_list
    random.shuffle(dataset)

    # Calculate the number of samples
    num_total_samples = len(dataset)

    # Create the train, validation, and test masks
    train_mask, val_mask, test_mask = create_masks(num_total_samples, args.train_ratio, args.valid_ratio, args.test_ratio)

    assert train_mask.sum()+val_mask.sum()+test_mask.sum() == num_total_samples

    train_set = [data for data, mask in zip(dataset, train_mask) if mask]
    val_set = [data for data, mask in zip(dataset, val_mask) if mask]
    test_set = [data for data, mask in zip(dataset, test_mask) if mask]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    model = GNN_GraphGIN(10, num_classes)
    model.to(device)

    if args._eval:
        print(f"Training an GNN_eval with b = {args.b}")

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    losses = {'train':[], 'val': []}

    for epoch in range(0, args.epoch):
        model.train()

        # Use pytorch-geometric batching method
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            if args._eval:  ##PERFORM RANDOM EDGE DROPPING
                sample_adj = Randomized_Edge_Dropping(data.edge_index, b=args.b, device=device)
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
            store_checkpoint("GNN", dataset_name, model, train_acc, val_acc, test_acc, best_epoch, args._eval)

        # Early stopping
        if epoch - best_epoch > args.early_stopping:
            break

    model = load_best_model(best_epoch,"GNN", dataset_name, model, True, args._eval)

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
        val_acc_full = evaluate(out, eval_data.y)

        if args.compute_lambda:
            results = []
            Lambda = 0
            for d in np.linspace(0, 0.5, 10):
                sample_adj = Randomized_Edge_Dropping(eval_data.edge_index,b=d,device=device)
                sample_edge_weights = torch.ones(sample_adj.size(1)).to(device)
                out = model(eval_data.x, sample_adj, eval_data.batch, sample_edge_weights)
                val_acc = evaluate(out, eval_data.y)
                # Log val_acc for each b
                results.append((f"Drop Prob {d}", val_acc))
                Lambda += val_acc
            # Log total Lambda
            results.append(('Lambda', Lambda/10))

            # Write results to a CSV file
            save_dir ="results"
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, f"{dataset_name}_lambda.csv"), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(results)

    print(f"final train_acc:{train_acc}, val_acc: {val_acc_full}, test_acc: {test_acc}")
    if args.compute_lambda:
        print(f"Lambda Value on the Validation Set {Lambda/10}")

    store_checkpoint("GNN", dataset_name, model, train_acc, val_acc_full, test_acc, -1, args._eval)

def evaluate_gnns(args):
    if args.save_dir == "default":
        dataset_name = "Graph_Motifs_Plus"
    if args.save_dir != "default":
        dataset_name = f"Graph_Motifs_Plus_{args.save_dir}"
    
    print(f"===== Working on {dataset_name} =====")
    if args._eval:
        print("==== The GNN_eval model for this dataset is the following ====")
        model, checkpoint = model_selector("GNN",
                                        dataset_name,
                                        pretrained=True,
                                        return_checkpoint=True,
                                        _eval=True,
                                        device=device
                                    )
        print(model)
        model.eval()
        model.to(device)
        model_name = "eval"

    if not args._eval:
        print("==== The vanilla GNN model for this dataset is the following ====")
        model, checkpoint = model_selector("GNN",
                                        dataset_name,
                                        pretrained=True,
                                        return_checkpoint=True,
                                        _eval=False,
                                        device=device
                                        )
        print(model)
        model.eval()
        model.to(device)
        model_name = "gnn"

    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=['Unimp Probability',
                                       'Model Total Accuracy',
                                       'Model on circular_ladder',
                                       'Model on complete',
                                       'Model on grid',
                                       'Model on lollipop',
                                       'Model on wheel',
                                       'Model on star',
                                       'Model on cycle'
                                       ], dtype=object)

    for unimp in np.linspace(0, 1.0, 11):

        num_nodes_motif = 6
        circular_ladder = [create_motifs_with_varying_unimp("circular_ladder", unimp, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
        complete = [create_motifs_with_varying_unimp("complete", unimp, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
        grid = [create_motifs_with_varying_unimp("grid", unimp, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
        lollipop = [create_motifs_with_varying_unimp("lollipop", unimp, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
        wheel = [create_motifs_with_varying_unimp("wheel", unimp, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
        star = [create_motifs_with_varying_unimp("star", unimp, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
        cycle = [create_motifs_with_varying_unimp("cycle", unimp, num_nodes_motif, args.num_nodes_base) for i in range(args.num_samples)]
        
        circular_ladder_data_list = create_data_list(circular_ladder, 0)
        complete_data_list = create_data_list(complete, 1)
        grid_data_list = create_data_list(grid, 2)
        lollipop_data_list = create_data_list(lollipop, 3)
        wheel_data_list = create_data_list(wheel, 4)
        star_data_list = create_data_list(star, 5)
        cycle_data_list = create_data_list(cycle, 6)
        # Combine the data lists into a single dataset
        dataset = circular_ladder_data_list+complete_data_list+grid_data_list+lollipop_data_list\
                +wheel_data_list+star_data_list+cycle_data_list       
        random.shuffle(dataset)
        dataset_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        # Evaluate train
        with torch.no_grad():
            eval_data = next(iter(dataset_loader)) # Loads all test samples
            eval_data.to(device)

            out = model(eval_data.x, eval_data.edge_index, eval_data.batch, eval_data.edge_weights)
            gnn_total_acc = evaluate(out, eval_data.y)
            gnn_circular_ladder_acc = evaluate_specific(out, eval_data.y, 0)
            gnn_complete_acc = evaluate_specific(out, eval_data.y, 1)
            gnn_grid_acc = evaluate_specific(out, eval_data.y, 2)
            gnn_lollipop_acc = evaluate_specific(out, eval_data.y, 3)
            gnn_wheel_acc = evaluate_specific(out, eval_data.y, 4)
            gnn_star_acc = evaluate_specific(out, eval_data.y, 5)
            gnn_cycle_acc = evaluate_specific(out, eval_data.y, 6)

        results_df = results_df.append({
                'Unimp Probability': unimp, 
                'Model Total Accuracy': gnn_total_acc, 
                'Model on circular_ladder':gnn_circular_ladder_acc, 
                'Model on complete':gnn_complete_acc, 
                'Model on grid':gnn_grid_acc, 
                'Model on lollipop':gnn_lollipop_acc, 
                'Model on wheel':gnn_wheel_acc, 
                'Model on star':gnn_star_acc, 
                'Model on cycle':gnn_cycle_acc, 
            }, ignore_index=True)
        print(results_df, flush=True)
    
    # Write results to a CSV file
    save_dir ="results"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    results_df.to_csv(os.path.join(save_dir, f'{dataset_name}_{model_name}_Unimp.csv'),index=False)

    lambda_true = results_df['Model Total Accuracy'].mean()
    
    print(f"Lambda_true: {lambda_true}")