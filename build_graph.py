"""Generate a random directed graph via an adjacency matrix, build a spanning tree with its edges, add random edges until it hits an edge count, and output the resulting graph in the form of an adjacency matrix."""

import numpy as np
from graph import Edge, Node

def random_adj_matrix(num_roads: int, min_weight: int = 1, max_weight: int = 10) -> np.array:
    random_matrix = np.randint(min_weight, max_weight, size=(num_roads, num_roads))
    return random_matrix

import numpy as np
import random

def make_spanning_tree(num_roads: int) -> np.array:
    # Step 1: Initialize an empty adjacency matrix
    adj_matrix = np.zeros((num_roads, num_roads), dtype=int)

    # Step 2: Create a connected path (linear chain) to ensure all nodes are reachable
    for i in range(num_roads - 1):
        adj_matrix[i][i + 1] = 1
        adj_matrix[i + 1][i] = 1

    # Step 3: Randomly add additional edges to make the graph more complex
    num_additional_edges = num_roads  # You can adjust this to control the sparsity
    for _ in range(num_additional_edges):
        u, v = random.sample(range(num_roads), 2)
        adj_matrix[u][v] = 1
        adj_matrix[v][u] = 1

    # Step 4: Generate the spanning tree using BFS
    spanning_tree = np.zeros((num_roads, num_roads), dtype=int)
    root = random.randint(0, num_roads - 1)
    queue = [root]
    visited = set()
    visited.add(root)

    while queue:
        current = queue.pop(0)
        for neighbor in range(num_roads):
            if adj_matrix[current][neighbor] == 1 and neighbor not in visited:
                spanning_tree[current][neighbor] = 1
                spanning_tree[neighbor][current] = 1  # Maintain undirected structure
                visited.add(neighbor)
                queue.append(neighbor)

    return spanning_tree


def make_graph(adj: list[list[int]]):
    adj_list, correspondence = preprocess(adj)
    return connect(adj_list, correspondence)
    

def connect(adj_list: dict[int, int], correspondence: dict[tuple[int, int], Node]) -> tuple[list[Node], set[Edge]]:
    
    edges = set()
    
    for key, value in adj_list.items():
        for neighbor in value:
            for node in adj_list[neighbor]:
                from_node = correspondence[key, neighbor]
                to_node = correspondence[neighbor, node]
                edges.add(Edge(from_node, to_node))
                from_node.add_outgoing(to_node)
                to_node.add_incoming(from_node)
    
    return list(correspondence.values()), edges
    
def preprocess(adj_matrix: np.array) -> tuple[dict[int, int], dict[tuple[int, int], Node]]:
    
    correspondence = {}
    
    n = adj_matrix.shape[0]
    
    adj_list = {i: [] for i in range(n)}
    for i, j in zip(*np.where(adj_matrix != 0)):
        adj_list[i].append(j)
        correspondence[(i, j)] = Node()
    
    return adj_list, correspondence