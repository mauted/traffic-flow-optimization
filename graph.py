from __future__ import annotations
from collections import namedtuple
import numpy as np

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

    def __init__(self, num_nodes, num_edges):
        pass