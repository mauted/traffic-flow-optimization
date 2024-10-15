import networkx as nx

import matplotlib.pyplot as plt

"""
For now, this program simply creates a 2D grid graph and prints and displays its
nodes and edges, based on their relative coordinates.

Each internal node (degree 4) corresponds to an intersection, while each 
external node corresponds to an entry/exit point.
"""

INTERNAL_WIDTH = 2
INTERNAL_HEIGHT = 5

width = INTERNAL_WIDTH + 2
height = INTERNAL_HEIGHT + 2

# Create a grid graph
G = nx.grid_2d_graph(width, height)

# Remove the outer rim of edges
for i in range(width - 1):
  G.remove_edge((i, 0), (i + 1, 0))
  G.remove_edge((i, height - 1), (i + 1, height - 1))
for i in range(height - 1):
  G.remove_edge((0, i), (0, i + 1))
  G.remove_edge((width - 1, i), (width - 1, i + 1))

# Remove corner nodes
G.remove_node((0, 0))
G.remove_node((0, height - 1))
G.remove_node((width - 1, 0))
G.remove_node((width - 1, height - 1))

# Draw the graph
pos = dict((n, n) for n in G.nodes())
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", 
        font_size=10, font_weight="bold")

# Print the graph as the set of nodes and edges
print("Nodes of the graph:", G.nodes())
print("Edges of the graph:", G.edges())

# Show the plot
plt.show()
