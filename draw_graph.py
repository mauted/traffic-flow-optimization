import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph import make_spanning_tree

def draw_graph_from_matrix(adj_matrix):
    # Create a directed graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, arrows=True)
    # nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True)
    plt.title("Spanning Tree")
    plt.show()

def draw_graph_from_list(adj_list: dict):
    
    G = nx.DiGraph()
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    pos = nx.spring_layout(G) 
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, arrows=True)
    plt.title("Spanning Tree")
    plt.show()


if __name__ == "__main__":
    num_roads = 100
    spanning_tree_matrix = make_spanning_tree(num_roads)
    draw_graph_from_matrix(spanning_tree_matrix)