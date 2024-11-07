"""Generate a random directed graph via an adjacency matrix, build a spanning tree with its edges, add random edges until it hits an edge count, and output the resulting graph in the form of an adjacency matrix."""

from collections import namedtuple
import numpy as np
from graph import Graph, Edge, Node

def random_adj_matrix(num_roads: int, min_weight: int = 1, max_weight: int = 10) -> np.array:
    random_matrix = np.randint(min_weight, max_weight, size=(num_roads, num_roads))
    return random_matrix

def make_spanning_tree(num_roads: int) -> np.array:
    adj_matrix = np.randint(0, 2, size=(num_roads, num_roads))
    np.fill_diagonal(adj_matrix, 0)

    # Perform BFS on a random root node to generate a spanning tree
    root = np.random.randint(num_roads)
    queue = [root]
    visited = set()
    visited.add(root)
    while queue:
        current = queue.pop(0)
        for i in range(num_roads):
            if adj_matrix[current][i] == 1 and i not in visited:
                adj_matrix[current][i] = 0
                visited.add(i)
                queue.append(i)
    return adj_matrix
    

def make_graph(adj: list[list[int]]) -> Graph:
    """
    Makes a graph object based on the adjacency matrix of a weighted, directed graph.
    This method flips the definition of nodes and edges, such that nodes in the original graph G become edges in G', and edges in G become nodes in G'.
    onodes = nodes in G, oedges = edges in G
    nnodes = nodes in G', nedges = edges in G'
    """
    

def connect(adj_list: dict[int, int], correspondence: dict[tuple[int, int], Node]) -> tuple[list[Node], set[Edge]]:
    
    edges = set()
    
    for key, value in adj_list.items():
        for neighbor in value:
            for node in adj_list[neighbor]:
                from_node = correspondence[key, neighbor]
                to_node = correspondence[neighbor, node]
                edges.add(Edge(from_node, to_node))
                from_node.add_outgoing(to_node)
    
    return correspondence.values(), edges
    
def preprocess(adj_matrix: np.array) -> tuple[dict[int, int], dict[tuple[int, int], Node]]:
    
    correspondence = {}
    
    n = adj_matrix.shape[0]
    
    adj_list = {i: [] for i in range(n)}
    for i, j in zip(*np.where(adj_matrix != 0)):
        adj_list[i].append(j)
        correspondence[(i, j)] = Node()
    
    return adj_list, correspondence