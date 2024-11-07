from graph import Node
from build_graph import connect, preprocess
import numpy as np
import pytest

adj_matrix_1 = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
])
adj_matrix_2 = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
])
adj_matrix_3 = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0]
])
adj_matrix_4 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
adj_matrix_5 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
adj_matrix_6 = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 1, 0]
])
adj_matrix_7 = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])
adj_matrix_8 = np.array([
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])
adj_list_1, corr_1 = preprocess(adj_matrix_1)
adj_list_2, corr_2 = preprocess(adj_matrix_2)
adj_list_3, corr_3 = preprocess(adj_matrix_3)
adj_list_4, corr_4 = preprocess(adj_matrix_4)
adj_list_5, corr_5 = preprocess(adj_matrix_5)
adj_list_6, corr_6 = preprocess(adj_matrix_6)
adj_list_7, corr_7 = preprocess(adj_matrix_7)
adj_list_8, corr_8 = preprocess(adj_matrix_8)

nodes_1, edges_1 = connect(adj_list_1, corr_1)
nodes_2, edges_2 = connect(adj_list_2, corr_2)
nodes_3, edges_3 = connect(adj_list_3, corr_3)
nodes_4, edges_4 = connect(adj_list_4, corr_4)
nodes_5, edges_5 = connect(adj_list_5, corr_5)
nodes_6, edges_6 = connect(adj_list_6, corr_6)
nodes_7, edges_7 = connect(adj_list_7, corr_7)
nodes_8, edges_8 = connect(adj_list_8, corr_8)

@pytest.fixture(autouse=True)
def reset():
    Node.reset()

def test_preprocess():
    
    assert adj_list_1 == {
        0: [1],
        1: [2],
        2: []
    }
    assert list(corr_1.keys()) == [(0, 1), (1, 2)]

    assert adj_list_2 == {
        0: [],
        1: [],
        2: []
    }
    assert list(corr_2.keys()) == []

    assert adj_list_3 == {
        0: [1, 2],
        1: [2],
        2: [0]
    }
    assert list(corr_3.keys()) == [(0, 1), (0, 2), (1, 2), (2, 0)]

    assert adj_list_4 == {
        0: [0],
        1: [1],
        2: [2]
    }
    assert list(corr_4.keys()) == [(0, 0), (1, 1), (2, 2)]

    assert adj_list_5 == {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1]
    }
    assert list(corr_5.keys()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    assert adj_list_6 == {
        0: [1, 2],
        1: [2],
        2: [1]
    }
    assert list(corr_6.keys()) == [(0, 1), (0, 2), (1, 2), (2, 1)]

    assert adj_list_7 == {
        0: [1],
        1: [2],
        2: [0]
    }
    assert list(corr_7.keys()) == [(0, 1), (1, 2), (2, 0)]

    assert adj_list_8 == {
        0: [1, 3],
        1: [2],
        2: [0],
        3: [2]
    }
    assert list(corr_8.keys()) == [(0, 1), (0, 3), (1, 2), (2, 0), (3, 2)]

def test_connect():
    
    assert len(edges_8) == 6
    assert (nodes_8[0], nodes_8[2]) in edges_8
    assert (nodes_8[1], nodes_8[4]) in edges_8
    assert (nodes_8[2], nodes_8[3]) in edges_8
    assert (nodes_8[3], nodes_8[0]) in edges_8
    assert (nodes_8[3], nodes_8[1]) in edges_8
    assert (nodes_8[4], nodes_8[3]) in edges_8
    
    adj_list = {
        0: [1, 2, 3, 4],
        1: [0, 2, 4],
        2: [0, 1, 3],
        3: [0, 1, 2, 4],
        4: [0, 1, 3]
    }
    corr = {}
    for key, val in adj_list.items():
        for neighbor in val:
            corr[(key, neighbor)] = Node() 
    nodes, edges = connect(adj_list, corr)
    assert (nodes[0], nodes[4]) in edges
    assert (nodes[9], nodes[13]) in edges
    assert (nodes[14], nodes[2]) in edges
    assert (nodes[8], nodes[6]) in edges
    assert (nodes[12], nodes[8]) in edges
    assert (nodes[7], nodes[2]) in edges