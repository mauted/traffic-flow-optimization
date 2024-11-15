import numpy as np
from graph import make_spanning_tree
import pytest
from collections import deque

# Import the function here if it's in a different module
# from your_module import make_spanning_tree

NUM_NODES = 5
NUM_EDGES = 10

def test_spanning_tree_shape():
    """Test that the output adjacency matrix has the correct shape."""
    adj_matrix = make_spanning_tree(NUM_NODES, NUM_EDGES)
    assert adj_matrix.shape == (NUM_NODES, NUM_NODES), "Adjacency matrix has incorrect shape."

def test_connectivity():
    """Test that the spanning tree is connected (all nodes are reachable)."""
    adj_matrix = make_spanning_tree(NUM_NODES, NUM_NODES)
    
    # Perform BFS/DFS to check connectivity
    visited = set()
    q = deque([0])  # Start BFS from node 0
    
    while q:
        node = q.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in range(NUM_NODES):
            if adj_matrix[node][neighbor] == 1 and neighbor not in visited:
                q.append(neighbor)
    
    assert len(visited) == NUM_NODES, "The spanning tree is not fully connected."

