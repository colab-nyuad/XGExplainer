# XGExplainer: Experiments on Node and Graph Classification

This directory contains the code and datasets for node and graph classification experiments.

For the experiment of explanations for node and graph classification, we customize the experiment setting and implementation of GNNExplainer taken from [ MixupExplainer: Generalizing Explanations for Graph Neural
Networks with Data Augmentation ](https://github.com/jz48/MixupExplainer), and PGExplainer taken from [ \[Re\] Parameterized Explainer for Graph Neural Networks](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks#re-parameterized-explainer-for-graph-neural-networks).

- Note that by default, the experiment will use the pretrained models that are saved in `ExplanationEvaluation/models/pretrained/GNN/{dataset_name}`. If you wish to use a newly trained model, move the trained model from `checkpoints/GNN/{dataset_name}`. A standard GNN will be saved in the directory `dataset_name` while GNN<sub>eval</sub> is saved in `dataset_name/GNN_EVAL`. 

- We used two different conda environment to run GNNExplainer and PGExplainer. This is due to the inability to use the original implementation of GNNExplaine due to a depricated functionality of Pytorch Geometric. Please prepare the following two different environments for reproducing XGExplainer<sub>+GNNE</sub> and  XGExplainer<sub>+PGE</sub>, respectively.

**Note**: 
Due to the lack of absolute reproducibility in PyTorch, even when using the same random seed, there is no assurance of obtaining identical results (refer to [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)). Consequently, there may be slight deviations from the findings reported in the paper.

## Requirements
### For XGExplainer<sub>+GNNE</sub> (**Env 1**)
The code was tested with the following packages:
```
python==3.8.0
cudatoolkit==11.3.1
pytorch==1.12.1
torch-geometric==2.1.0.post1
pytorch-sparse=0.6.15
matplotlib==3.5.3
seaborn==0.12.2
ipython==7.31.1
```

- Install python==3.8.0
- Install `pytorch` and its dependencies following [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). We installed them by `conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch`. 
- For `pytorch geometric` and its dependencies, install following [https://pypi.org/project/torch-geometric/2.1.0.post1/](https://pypi.org/project/torch-geometric/2.1.0.post1/). We installed them by `pip install torch-geometric==2.1.0.post1` *I think this was the way I installed
- Install torch-sparse `conda install pytorch-sparse=0.6.15 -c pyg`
- Install the remaining packages.

### For XGExplainer<sub>+PGE</sub> (**Env 2**)

The code was tested with the following packages:
```
python==3.7.13
cudatoolkit==11.3.1
pytorch==1.11.0
pyg==2.0.4
dgl-cuda11.1==0.9.1
matplotlib==3.5.3
seaborn==0.12.2
ipython==7.31.1
```
- Install `pytorch` and its dependencies following [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). We installed them by `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`. 
- For `pytorch geometric` and its dependencies, install following [https://pypi.org/project/torch-geometric/2.0.4/](https://pypi.org/project/torch-geometric/2.0.4/). We installed them by `conda install pyg=2.0.4 -c pyg`
- For `dgl`, install following the instruction at [https://anaconda.org/dglteam/dgl-cuda11.1](https://anaconda.org/dglteam/dgl-cuda11.1). We installed the package using `conda install -c dglteam dgl-cuda11.1`.
- Install `matplotlib`, `seaborn`, and `ipython`. 


# Synthetic Controlled Experiments 
Use **Env 2** for the following experiments.
## Generalizing Abstract Structures
This experiment evaluates the robustness of GNN<sub>eval</sub> as an evaluator GNN when graphs $G^p_X$ comes from a different distribution while maintaining the important structures. This experiment performs graph classification on Graph-Motifs dataset.

To train GNN and GNN<sub>eval</sub>, use the following command. Use the `--_eval` flag to specify that you are training GNN<sub>eval</sub>. Otherwise, do not include the flag.
```
python run_graph_motifs.py --train_model --_eval
```

For evaluating GNN and GNN<sub>eval</sub> on the OOD set, use the following command:
```
python run_graph_motifs.py --evaluate_GNN --_eval
```

The result will be stored in `results/Graph_Motifs_{gnn/eval}_{IID/OOD}_Accuracy.csv` files.

## Effect of Masking Unimportant Edges
This experiment tests evaluator GNN's ability to perform predictions when important subgraphs are fed by masking out the unimportant edges. This experiment performs graph classification on Graph-Motifs+ dataset.

For training GNN and GNN<sub>eval</sub>, use the following command, with the flag `--_eval` to specify that you are training a GNN<sub>eval</sub>.
```
python run_graph_pruning.py --train_model --_eval
```

For evaluating GNN and GNN<sub>eval</sub> with different $P(unimp)$, run the following command.
```
python run_graph_pruning.py --evaluate_GNN --_eval
```
The result will be stored in `results/Graph_Motifs_Plus_{gnn/eval}_Unimp.csv` files.


## Choosing the Hyperparameter $b$
This experiment examines the effect of changing the hyperparameter $b$, which is the probability to drop an edge during training of GNN<sub>eval</sub>

For training GNN<sub>eval</sub> with different hyperparamer $b$, specify the value using the argument `--b`. Remember to use the flag `--compute_lambda` to get $\lambda$. By using the `--save_dir` argument, you can save your model in `checkpoints/GNN/Graph_Motifs_Plus_{save_dir}` directory to avoid overlapping.

For example, to train GNN<sub>eval</sub> with $b = 0.05$, use the following:
```
python run_graph_pruning.py --train_model --compute_lambda --_eval --b=0.05 --save_dir=0.05
```

$\lambda$ is stored in the file `results/Graph_Motifs_Plus_{b}_lambda.csv`

You can calculate $\lambda$ for GNN as well, using the following (or setting $b=0$ using the above format):
```
python run_graph_pruning.py --train_model --compute_lambda
```

To evaluate the performance of GNN<sub>eval</sub> with different hyperparameter $b$, you can run the following command. We use $b = 0.05$ as an example.
```
python run_graph_pruning.py --evaluate_GNN --_eval --save_dir=0.05
```

For GNN<sub>eval</sub>, the result will be stored in `results/Graph_Motifs_Plus_{b}_eval_Unimp.csv` files. To get $\lambda_{true}$, which would also be printed out when you run the program, simply calculate the average of the column 'Model Total Accuracy'.

The $\lambda_{true}$ value for GNN is the average of the column 'Model Total Accuracy' on the file `results/Graph_Motifs_Plus_gnn_Unimp.csv`

# Explanation on Node and Graph Classification

Using the following commands, you can run the experiment of XGExplainer on Node and Graph Classification datasets. The experiment uses pretrained GNN<sub>eval</sub> models.
The command line arguments are as follows:
```
--dataset: the dataset we run the experiment for. Options: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
--all_seeds: whether to evaluate on all (10) seeds. Use this flag to conduct, otherwise, do not include this argument. If not used, XGExplainer will be evaluated on seed 0 only.
--run_qual: whether to get a qualitative image of the explanation. Use this flag to conduct, otherwise, do not include this argument.
--explainer: Choose the base explainer as either pgexplainer or gnnexplainer.
```
Make sure to use **Env 1** for XGExplainer<sub>+GNNE</sub> and **Env 2** for XGExplainer<sub>+PGE</sub>.

For example, to get the result of XGExplainer<sub>+PGE</sub> on bashapes dataset with all the seeds:
```
python run_xgexplainer.py --explainer=pgexplainer --dataset=bashapes --all_seeds
```

The result will be stored in `results/XGExplainer_{PG/GNN}_{dataset}.csv` file. (Dataset alias is the following: syn1=bashapes, syn2=bacommunity, syn3=treecycles, syn4=treegrids, ba2=ba2motifs, mutag=mutag). 

**Warning**: The creation of Mutag dataset takes a lot of memory. Be sure to allocate enough memory for the job.

If you wish to retrain GNN<sub>eval</sub>, you may run the following:
```
python train_GNN_eval.py --dataset bashapes
```
Where `--dataset` flag follows the same utility as the explanation command.

For hyperparameter tuning, we performed grid search of hyperparameters of the baseline PGExplainer for the rang specified in `hyper_range.txt`.

# Results

## Generalizing Abstract Structures

For this experiment, the pretrained models achieve this performance on the IID and OOD test sets.

|                 | GNN               |                   | GNN_eval          |                   |
|-----------------|-------------------|-------------------|-------------------|-------------------|
| Class           | IID Test Accuracy | OOD Test Accuracy | IID Test Accuracy | OOD Test Accuracy |
| Circular Ladder |        1.00       |        1.00       |        1.00       |        1.00       |
|     Complete    |        1.00       |        0.40       |        1.00       |        1.00       |
|       Grid      |        1.00       |        1.00       |        0.90       |        1.00       |
|     Lollipop    |        1.00       |        0.49       |        0.95       |        0.60       |
|      Wheel      |        1.00       |        0.80       |        1.00       |        1.00       |
|       Star      |        1.00       |        0.40       |        1.00       |        1.00       |
|      Cycle      |        1.00       |        1.00       |        1.00       |        1.00       |
|  Total Accuracy |        1.00       |        0.73       |        0.98       |        0.94       |

## Effect of Masking Unimportant Edges

For this experiment, the pretrained models achieved the following performance.

| Unimp Probability ($\rho$) | GNN Total Accuracy | GNN<sub>eval</sub> Total Accuracy |
|-------------------|--------------------|-------------------------|
|                 0 |               0.57 |                    0.86 |
|               0.1 |               0.57 |                    0.84 |
|               0.2 |               0.57 |                    0.87 |
|               0.3 |               0.57 |                    0.92 |
|               0.4 |               0.58 |                    0.94 |
|               0.5 |               0.58 |                    0.95 |
|               0.6 |               0.61 |                    0.97 |
|               0.7 |               0.67 |                    0.97 |
|               0.8 |               0.74 |                    0.97 |
|               0.9 |               0.84 |                    0.95 |
|                 1 |               0.98 |                    0.88 |

## Choosing the Hyperparameter $b$

We obtained the following result for the hyperparameter tuning experiment.

| b    |$\lambda$ | $\lambda_{true}$ |
|------|----------------|-------|
|    0 |           0.33 |  0.66 |
| 0.01 |           0.36 |  0.68 |
| 0.05 |           0.44 |  0.86 |
|  0.1 |           0.51 |  0.93 |
|  0.2 |           0.45 |  0.72 |
|  0.3 |           0.36 |  0.33 |
|  0.4 |           0.30 |  0.29 |
|  0.5 |           0.14 |  0.14 |

## Explanation on Node and Graph Classification

For obtaining explanations on node and graph classification tasks, we achieved the following result.

|             | Node Classification |               |               |               | Graph Classification |               |
|-------------|:-------------------:|:-------------:|:-------------:|:-------------:|:--------------------:|:-------------:|
|             |      BA-Shapes      |  BA-Community |  Tree-Cycles  |   Tree-Grids  |      BA-2motifs      |     MUTAG     |
| XGExplainer<sub>+GNNE</sub> |    0.958 ± 0.001    | 0.839 ± 0.001 | 0.749 ± 0.002 | 0.741 ± 0.003 |     0.708 ± 0.011    | 0.696 ± 0.001 |
| XGExplainer<sub>+PGE</sub> |    1.000 ± 0.000    | 0.993 ± 0.002 | 0.944 ± 0.003 | 0.940 ± 0.014 |     0.971 ± 0.014    | 0.919 ± 0.032 |0p-