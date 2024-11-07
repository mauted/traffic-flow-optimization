"""Generate a random directed graph via an adjacency matrix, build a spanning tree with its edges, add random edges until it hits an edge count, and output the resulting graph in the form of an adjacency matrix."""

from collections import namedtuple
import numpy as np
from graph import Graph, Edge, Node

Road = namedtuple('Road', ['capacity', 'time'])

def random_adj_matrix(num_roads: int, min_weight: int = 1, max_weight: int = 10) -> np.array:
    random_matrix = np.randint(min_weight, max_weight, size=(num_roads, num_roads))
    return random_matrix

def make_random_graph(num_roads: int) -> Graph:
    """
    Makes a randomized fully connected spanning tree with given number of nodes. This method will generate adjacency matrix,
    and then ensure that the tree is fully connected by adding edges until it is connected.
    """
    adj_matrix = np.array(dtype=Road, shape=(num_roads, num_roads))
    

def make_graph(adj_matrix: np.array) -> Graph:
    """
    Makes a graph object based on the adjacency matrix of a weighted, directed graph.
    This method flips the definition of nodes and edges, such that nodes in the original graph G become edges in G', and edges in G become nodes in G'.
    onodes = nodes in G, oedges = edges in G
    nnodes = nodes in G', nedges = edges in G'
    """
    
    assert adj_matrix.shape[0] == adj_matrix.shape[1]

    # create the new nodes
    num_nnodes = np.count_nonzero(adj_matrix)
    nnodes = [Node() for _ in range(num_nnodes)]
    
    # establish connections based on the adjacency matrix
    