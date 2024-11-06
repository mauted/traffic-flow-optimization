from __future__ import annotations
from collections import namedtuple
import random 
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

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
        
    def __repr__(self):
        return str(self.id)
    

class Intersection:
    
    __COUNTER = 0

    def __init__(self, nodes: list[Node]): 
        self.id = Intersection.__COUNTER
        Intersection.__COUNTER += 1
        self.in_roads, self.out_roads, self.edges = self.construct_intersection(nodes)
    
    @staticmethod
    def construct_intersection(nodes: list[Node]) -> tuple[list[Node], list[Node], list[Edge]]:
        """
        Takes an input of the form [IN, OUT, BOTH,...] and returns a constructed intersection
        with the specified number of roads. Each road going into the intersection connects to each 
        road going out of the intersection.
        """
        in_roads = [Node() for _ in nodes.incoming]
        out_roads = [Node() for _ in nodes.outgoing]
        edges = [Edge(in_road, out_road) for in_road in in_roads for out_road in out_roads]

        return in_roads, out_roads, edges

        
class Graph: 
    
    def __init__(self, num_intersections, num_edges):
        self.nodes, self.edges, self.intersections = Graph._build_complete_network(num_intersections, num_edges)
        breakpoint()
        
    
    @staticmethod
    def _build_complete_network(num_intersections: int, num_edges: int) -> tuple[list[Node], list[Edge], list[Intersection]]:
        """
        Generates the complete road network, from a given number of edges.
        """
        pre_intersections = Graph._build_supergraph(num_intersections, num_edges)
        nodes = []
        edges = []
        intersections = []

        for pre_int in pre_intersections:
            intersection = Intersection(pre_int)
            intersections.append(intersection)

            in_roads = intersection.in_roads
            out_roads = intersection.out_roads
            inter_edges = intersection.edges

            nodes += in_roads + out_roads
            edges += inter_edges
        
        return nodes, edges, intersections
    
    @staticmethod
    def _build_supergraph(num_intersections: int, num_edges: int) -> list[Node]:
        """
        Constructs a randomly generated connected directed graph with num_nodes pre-intersection nodes
        and num_edges edges.
        """
        
        if num_intersections < 2:
            raise ValueError("Number of intersections must be at least 2")
        if num_edges < num_intersections - 1:
            raise ValueError("Number of edges must be at least num_nodes - 1 to ensure connectivity")
        
        adj_matrix = np.zeros((num_intersections, num_intersections))

        for i in range(num_intersections):
            for j in range(i + 1, num_intersections): 
                weight = np.random.randint(1, 11)
                adj_matrix[i][j] = weight
                adj_matrix[j][i] = weight 

        adj_matrix = np.random.randint(1, 11, size=(5, 5))
        mst = minimum_spanning_tree(adj_matrix)
        

        current_edges = num_intersections - 1
        while current_edges < num_edges:
            i, j = possible_edges.pop()
            pre_nodes[i].add_outgoing(pre_nodes[j])
            pre_nodes[j].add_incoming(pre_nodes[i])
            current_edges += 1

        return pre_nodes
    
if __name__ == "__main__":
    graph = Graph(3, 10)