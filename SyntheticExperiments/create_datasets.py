import networkx as nx
import random
import torch
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def remove_edges_with_probability(G, probability):
    """
    function to remove edges from networkx graph G with a probability
    """
    edges_to_remove = []
    for edge in G.edges():
        if random.random() < probability:
            edges_to_remove.append(edge)

    G.remove_edges_from(edges_to_remove)
    return G

def create_motifs_with_varying_unimp(motif_name, unimp, num_nodes_motif, num_nodes_base):
    if num_nodes_motif < 6:
        raise ValueError("Number of Nodes for the Motif needs to be Minimum of 6.")
    
    # 1.0 unimp -> contain all the unimp edges, 0 unimp -> exclude all the unimp edges
    G = nx.random_tree(n = num_nodes_base)
    G = remove_edges_with_probability(G, 1-unimp)
    
    if motif_name == "circular_ladder":
        M = nx.circular_ladder_graph(num_nodes_motif//2)
    elif motif_name == "complete":
        M = nx.complete_graph(num_nodes_motif)
    elif motif_name == "grid":
        n_rows = num_nodes_motif//2
        n_cols = num_nodes_motif // n_rows
        M = nx.grid_graph([n_rows, n_cols])
    elif motif_name == "lollipop":
        path_len = np.random.randint(2, num_nodes_motif // 2)
        M = nx.lollipop_graph(m=num_nodes_motif - path_len, n=path_len) 
    elif motif_name == "wheel":
        M = nx.wheel_graph(num_nodes_motif)
    elif motif_name == "star":
        M = nx.star_graph(num_nodes_motif - 1)
    elif motif_name == "cycle":
        M = nx.cycle_graph(num_nodes_motif)
    else:
        raise ValueError("The specified motif is not implemented")
    G = nx.disjoint_union(G,M)
    if random.random() >= 1-unimp:
        random_node_motif = random.choice(range(5)) # choose a random node from the house
        random_node = random.choice(range(num_nodes_base)) # choose a random node from the BA
        random_node_motif = num_nodes_base + random_node_motif # update the node index to the correct one
        G.add_edge(random_node_motif, random_node) # add the edge
    return G