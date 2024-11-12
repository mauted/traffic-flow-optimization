from __future__ import annotations
from collections import namedtuple
import numpy as np
from build_graph import make_spanning_tree, make_graph
import random

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
    
class Road:
    # vehicles
    # capacity
    # time
    pass

class Simulation:
    # Contains a graph that is immutable
    pass
    
class Graph:

    def __init__(self, num_nodes):
        tree = make_spanning_tree(num_nodes)
        self.nodes, self.edges = make_graph(tree)

    def generate_random_path(self) -> list[Node]:
        start, end = random.sample(self.nodes, 2)
        return self.dfs(start, end)

    def dfs(self, start: Node, end: Node) -> list[Node]:
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
