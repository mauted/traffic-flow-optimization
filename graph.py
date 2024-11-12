from __future__ import annotations
import numpy as np
from collections import namedtuple
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation import Road

Edge = namedtuple('Edge', ['start', 'end'])

class Node:
    
    __COUNTER = 0
    
    def __init__(self):
        
        self.id = Node.__COUNTER
        Node.__COUNTER += 1 
        self.incoming = [] 
        self.outgoing = []

    def add_incoming(self, other: Node):
        self.incoming.append(other)
        
    def add_outgoing(self, other: Node):
        self.outgoing.append(other)
        
    @classmethod 
    def reset(cls):
        cls.__COUNTER = 0
        
    def __repr__(self):
        return str(self.id)
    
class Graph:

    def __init__(self, num_nodes, num_edges):
        tree = make_spanning_tree(num_nodes, num_edges)
        self.nodes, self.edges = make_graph(tree)


# def generate_random_path(nodes) -> list[Node]:
#     start, end = random.sample(nodes, 2)
#     return dfs(start, end)

def generate_random_path(roads, node_to_road_dict) -> list['Road']:
    start, end = random.sample(roads, 2)
    # nodes = [road.node for road in roads]
    # node_to_road = dict(zip(nodes, roads))
    path = dfs(start.node, end.node)

    return [node_to_road_dict[node] for node in path]

def dfs(start: Node, end: Node) -> list[Node]:
    """
    Returns a path from start to end using depth-first search.
    """
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        if node == end:
            return path  # Return the path if end node is found

        # Explore neighbors by extending the current path
        for neighbor in node.outgoing:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return []  # Return an empty list if no path is found


def random_adj_matrix(num_roads: int, min_weight: int = 1, max_weight: int = 10) -> np.array:
    random_matrix = np.randint(min_weight, max_weight, size=(num_roads, num_roads))
    return random_matrix


def make_spanning_tree(num_roads: int, num_edges: int) -> np.array:
    # Step 1: Initialize an empty adjacency matrix
    adj_matrix = np.zeros((num_roads, num_roads), dtype=int)

    # Step 2: Create a connected path (linear chain) to ensure all nodes are reachable
    for i in range(num_roads - 1):
        adj_matrix[i][i + 1] = 1
        adj_matrix[i + 1][i] = 1

    # Step 3: Randomly add additional edges to make the graph more complex
    if num_edges - num_roads > 0:
        num_additional_edges = num_edges - num_roads 
    else:
        num_additional_edges = 0
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

    # Step 5: Add some additional uni/bidirectional edges to the spanning tree
    remaining_extra_edges = num_roads // 2
    while remaining_extra_edges > 0:
        u, v = random.sample(range(num_roads), 2)
        added = False
        if spanning_tree[u][v] == 0:
            spanning_tree[u][v] = 1
            added = True
        if np.random.rand() < 0.5 and spanning_tree[v][u] == 0:
            spanning_tree[v][u] = 1
            added = True
        if added:
            remaining_extra_edges -= 1

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