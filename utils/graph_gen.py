import networkx as nx
import numpy as np
import random

def get_neighbors(n, graph_type):
    """
    Génère une liste d'adjacence pour n agents selon la topologie demandée.
    Garantit que chaque nœud a au moins 1 voisin pour satisfaire Target-UCB.
    """
    graph_type = graph_type.lower().replace("-", "_")
    
    if graph_type == "clique":
        G = nx.complete_graph(n)
        
    elif graph_type == "loop":
        G = nx.cycle_graph(n)
        
    elif graph_type == "chain":
        G = nx.path_graph(n)
        
    elif graph_type == "random":
        # Erdős-Rényi (p=0.5). On boucle pour garantir un graphe connecté (pas de nœud isolé)
        connected = False
        while not connected:
            G = nx.erdos_renyi_graph(n, p=0.5)
            # Si un nœud est isolé, les composants connectés seront > 1 ou le min_degree sera 0
            if all(d > 0 for _, d in G.degree()):
                connected = True
                
    elif graph_type == "small_world":
        # Modèle Barabási-Albert (m=1 voisin minimum par nœud ajouté)
        connected = False
        while not connected:
            G = nx.barabasi_albert_graph(n, m=1)
            if all(d > 0 for _, d in G.degree()):
                connected = True
    else:
        raise ValueError(f"Type de graphe inconnu : {graph_type}")
        
    # Extraction propre des voisins sous forme de dictionnaire de listes {i: [voisins]}
    adj = {i: list(G.neighbors(i)) for i in range(n)}
    
    # Sécurité ultime : si un nœud n'a vraiment pas de voisin, il s'auto-observe
    for i in range(n):
        if len(adj[i]) == 0:
            adj[i] = [i]
            
    return adj
