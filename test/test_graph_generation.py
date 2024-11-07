import numpy as np
from build_graph import make_spanning_tree
import pytest
from collections import deque

# Import the function here if it's in a different module
# from your_module import make_spanning_tree

def test_spanning_tree_shape():
    """Test that the output adjacency matrix has the correct shape."""
    num_nodes = 5
    adj_matrix = make_spanning_tree(num_nodes)
    assert adj_matrix.shape == (num_nodes, num_nodes), "Adjacency matrix has incorrect shape."

def test_connectivity():
    """Test that the spanning tree is connected (all nodes are reachable)."""
    num_nodes = 6
    adj_matrix = make_spanning_tree(num_nodes)
    
    # Perform BFS/DFS to check connectivity
    visited = set()
    queue = deque([0])  # Start BFS from node 0
    
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in range(num_nodes):
            if adj_matrix[node][neighbor] == 1 and neighbor not in visited:
                queue.append(neighbor)
    
    assert len(visited) == num_nodes, "The spanning tree is not fully connected."

