import networkx as nx
import matplotlib.pyplot as plt
import random
from old.graph import TrafficSupergraph

random.seed(0)

T = TrafficSupergraph.build_supergraph(10)
print(T.roads)
print([(edge.start, edge.end) for edge in T.edges])

for start in T.roads:
    print(f"Outgoing roads of road segment {start}: {start.outgoing}")
    print(f"Incoming roads of road segment {start}: {start.incoming}")

# Initialize directed graph in networkx
G = nx.DiGraph()

# Add nodes and edges
nodes = T.roads
edges = [(edge.start, edge.end) for edge in T.edges]

G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Check if T has predefined positions for nodes
if hasattr(T, 'positions'):
    pos = T.positions  # assume this is a dictionary of positions
else:
    pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(10, 8))
# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

# Draw edges with bidirectional curves
edges = G.edges()
nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=0.2')
nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=-0.2')

plt.title("Directed Graph T")
plt.show()
