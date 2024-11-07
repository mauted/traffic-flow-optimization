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

def test_num_edges():
    """Test that the spanning tree has exactly num_nodes - 1 edges."""
    num_nodes = 6
    adj_matrix = make_spanning_tree(num_nodes)
    num_edges = np.sum(adj_matrix)
    assert num_edges == 2 * (num_nodes - 1), f"Spanning tree should have {num_nodes - 1} edges, found {num_edges}."

def test_acyclicity():
    """Test that the generated graph is acyclic."""
    num_nodes = 6
    adj_matrix = make_spanning_tree(num_nodes)
    
    # Perform BFS/DFS to detect cycles
    visited = set()
    stack = [(0, -1)]  # (node, parent)
    
    while stack:
        node, parent = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in range(num_nodes):
            if adj_matrix[node][neighbor] == 1:
                if neighbor == parent:
                    continue  # Ignore the edge back to the parent
                if neighbor in visited:
                    pytest.fail("Cycle detected in spanning tree.")
                stack.append((neighbor, node))
    
    assert True  # Passed if no cycles were detected

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

def test_repeated_output():
    """Test that the function consistently produces valid spanning trees."""
    num_nodes = 6
    for _ in range(10):
        adj_matrix = make_spanning_tree(num_nodes)
        num_edges = np.sum(adj_matrix)
        
        # Check edge count and connectivity
        assert num_edges == 2 * (num_nodes - 1), "Invalid number of edges in spanning tree."
        
        # Check for connectivity
        visited = set()
        queue = deque([0])
        
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in range(num_nodes):
                if adj_matrix[node][neighbor] == 1 and neighbor not in visited:
                    queue.append(neighbor)
        
        assert len(visited) == num_nodes, "The spanning tree is not fully connected."

