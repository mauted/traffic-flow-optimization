import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from build_graph import make_spanning_tree

def draw_graph_from_matrix(adj_matrix):
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for better visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    plt.title("Spanning Tree")
    plt.show()


if __name__ == "__main__":
    num_roads = 64
    spanning_tree_matrix = make_spanning_tree(num_roads)
    draw_graph_from_matrix(spanning_tree_matrix)