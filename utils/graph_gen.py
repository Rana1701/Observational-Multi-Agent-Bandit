# utils/graph_gen.py
import networkx as nx
import numpy as np

def get_neighbors(n=20, graph_type='loop'):
    if graph_type == 'loop':
        G = nx.cycle_graph(n)
    elif graph_type == 'chain':
        G = nx.path_graph(n)
    elif graph_type == 'random':
        G = nx.erdos_renyi_graph(n, p=0.5)
    elif graph_type == 'small-world':
        G = nx.barabasi_albert_graph(n, m=2)
    
    adj = {i: list(G.neighbors(i)) for i in range(n)}
    return adj