from __future__ import annotations 
from enum import Enum
from collections import deque
import random

random.seed(0)


class TrafficLightAgent:

    def __init__(self):
        pass

class Vehicle:
    
    __COUNTER = 0

    def __init__(self, pos: RoadSegment, route: list[RoadSegment]):
        self.id = Vehicle.__COUNTER
        self.pos = pos 
        self.route = route
        Vehicle.__COUNTER += 1

class RoadSegment:
    
    __COUNTER = 0
    
    def __init__(self, duration: int = 30, capacity: int = 20):
        self.id = RoadSegment.__COUNTER
        RoadSegment.__COUNTER += 1
        self.outgoing = []
        self.incoming = []
        self.vehicles = []
        self.duration = duration
        self.capacity = capacity
    
    def add_incoming(self, road):
        self.incoming.append(road)
    
    def add_outgoing(self, road):
        self.outgoing.append(road)
    
    @classmethod 
    def reset(cls):
        cls.__COUNTER = 0
        
    def __eq__(self, other):
        if isinstance(other, RoadSegment):
            return self.id == other.id
        return False
    
    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return str(self.id)
    
    def move_car(self, car: Vehicle):
        pass
    
class Edge:
    
    __COUNTER = 0

    def __init__(self, start: RoadSegment, end: RoadSegment):
        self.id = Edge.__COUNTER
        Edge.__COUNTER += 1
        self.start = start
        self.end = end
        self.start.add_outgoing(self.end)
        self.end.add_incoming(self.start)
        # indicates whether this edge is active and allows vehicles to move on it
        self.active = True

    @classmethod 
    def reset(cls):
        cls.__COUNTER = 0
        
    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.id == other.id
        return False
    
    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return str(self.id)
    
    def move_car(self, car: Vehicle):
        pass

class PreIntersection:

__COUNTER = 0
    
    def __init__(self):
        self.id = PreIntersection.__COUNTER
        PreIntersection.__COUNTER += 1
        self.outgoing = []
        self.incoming = []
    
    def add_incoming(self, other):
        self.incoming.append(other)
    
    def add_outgoing(self, other):
        self.outgoing.append(other)
    
    @classmethod 
    def reset(cls):
        cls.__COUNTER = 0
        
    def __eq__(self, other):
        if isinstance(other, PreIntersection):
            return self.id == other.id
        return False
    
    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return str(self.id)
    
    def move_car(self, car: Vehicle):
        pass

class Intersection:

    __COUNTER = 0

    def __init__(self, template: PreIntersection): 
        self.id = Intersection.__COUNTER
        Intersection.__COUNTER += 1
        self.in_roads, self.out_roads, self.edges = self.construct_intersection(template)
        self.agent = TrafficLightAgent(self.edges)

    @staticmethod
    def construct_intersection(template: PreIntersection) -> tuple[list[RoadSegment], list[RoadSegment], list[Edge]]:
        """
        Takes an input of the form [IN, OUT, BOTH,...] and returns a constructed intersection
        with the specified number of roads. Each road going into the intersection connects to each 
        road going out of the intersection.
        """
        in_roads = template.in_roads
        out_roads = template.out_roads

        nodes = {node: RoadSegment() for node in set(in_roads + out_roads)}



        # constructing the nodes to the graph
        in_roads, out_roads = [], []
        for road in roads: 
            if road == RoadSegmentType.IN:
                in_roads.append(RoadSegment())            
            elif road == RoadSegmentType.OUT:
                out_roads.append(RoadSegment())
            else:
                out_roads.append(RoadSegment())
                in_roads.append(RoadSegment())

        # constructing the connections
        edges = []
        for in_road in in_roads:
            for out_road in out_roads:
                in_road.add_outgoing(out_road)
                out_road.add_incoming(in_road)
                edges.append(Edge(in_road, out_road))

        return in_roads, out_roads, edges

class RoadNetwork:
    """
    The domain of the environment for the RL simulations.
    """

    def __init__(self, nodes: list[RoadSegment], edges: list[Edge], vehicles: list[Vehicle]):
        self.nodes = nodes
        self.edges = edges 
        self.vehicles = vehicles

    @staticmethod
    def _build_supergraph(num_nodes: int) -> RoadNetwork:
        """
        Constructs a densely connected directed graph with num_nodes intersections (nodes).
        Each node is connected to multiple other nodes.
        """

        if num_nodes < 2:
            raise ValueError("Number of nodes must be at least 2")

        # Initialize nodes
        V = [RoadSegment() for _ in range(num_nodes)]
        E = []

        # Connect every node with multiple others
        for start in V:
            # Set a target number of edges each node will have
            target_outgoing = random.randint(2, num_nodes - 1)
            possible_ends = [node for node in V if node != start]
            
            # Create outgoing edges to reach the target connectivity
            for end in random.sample(possible_ends, target_outgoing):
                if end not in start.outgoing:
                    E.append(Edge(start, end))

        return RoadNetwork(V, E)
    
    @staticmethod
    def _build_planar_supergraph(num_nodes: int) -> RoadNetwork:
        """
        Constructs a planar directed graph with num_nodes intersections (nodes).
        Uses a mesh-like structure to ensure planarity.
        """

        if num_nodes < 2:
            raise ValueError("Number of nodes must be at least 2")

        # Initialize nodes
        V = [RoadSegment() for _ in range(num_nodes)]
        E = []

        # To ensure planarity, we create a grid or mesh structure
        side_length = int(num_nodes ** 0.5)  # Approximate side length for square mesh

        # Add edges to form a planar mesh
        for i in range(side_length):
            for j in range(side_length):
                current = i * side_length + j
                if current >= num_nodes:
                    break
                
                # Connect to the right neighbor if within bounds
                if j < side_length - 1 and current + 1 < num_nodes:
                    E.append(Edge(V[current], V[current + 1]))
                
                # Connect to the neighbor below if within bounds
                if i < side_length - 1 and current + side_length < num_nodes:
                    E.append(Edge(V[current], V[current + side_length]))

        # Optional: Add a few more edges randomly to enhance connectivity while ensuring planarity
        additional_edges = random.randint(1, num_nodes // 2)
        for _ in range(additional_edges):
            start, end = random.sample(V, 2)
            if end not in start.outgoing and start not in end.incoming:
                E.append(Edge(start, end))

        return RoadNetwork(V, E)
    
    @staticmethod
    def _build_complete_network():
        pass
        
    def generate_vehicle_paths(self, num_paths: int) -> list[list[RoadSegment]]:
        """
        Generates all paths for the vehicles of this traffic network
        """
        paths = []
        endpoints = self.generate_vehicle_endpoints(num_paths)
        for start, end in endpoints:
            paths.append(self.dfs(start, end))
        return paths

    def generate_vehicle_endpoints(self, num_paths: int) -> set[tuple[RoadSegment, RoadSegment]]: 
        """Generates a set of start and end nodes, representing a series of desired starting and ending points for the vehicles in the network."""
        paths = set()
        while len(paths) < num_paths:
            start, end = random.sample(self.nodes, 2)
            if (start, end) not in paths:
                paths.add((start, end))
        return paths
    
    def find_paths(self, paths: set[tuple[RoadSegment, RoadSegment]]) -> list[list[RoadSegment]]:
        """
        Finds a path with DFS through the traffic for each path in paths, and returns the list of roads representing the route that the car takes. 
        Note that the path is not necessarily optimal, nor does it need to be. 
        """
        
        found_paths = []
        
        for p in paths:
            
            parents = {}
            
            start, end = p
            frontier = deque()
            frontier.append(start)
            explored = set()
            
            while frontier: 
                node = frontier.pop()
                if node == end:
                    return self._traceback(node, parents)
                else: 
                    explored.add(node)
                    for succ in node.outgoing:
                        if succ not in explored: 
                            frontier.append(succ)
            
            found_paths.append(None)
        return found_paths

            
    def dfs(self, start: RoadSegment, end: RoadSegment) -> list[RoadSegment]:
        """Finds the path from start to end within the road network and returns the path as a list of roads"""
    
        parents = {}
            
        frontier = deque()
        frontier.append(start)
        explored = set()
        
        while len(frontier) != 0: 
            node = frontier.pop()
            if node == end:
                return self._traceback(node, parents)
            else: 
                explored.add(node)
                for succ in node.outgoing:
                    if succ not in explored and succ not in frontier:
                        parents[succ] = node
                        frontier.append(succ)
        
        return None
    
    def _traceback(self, node: RoadSegment, parents: dict[RoadSegment, RoadSegment]):
        """Traces back the path followed by the node in the parents dictionary"""
        path = []
        while node in parents: 
            path.append(node)
            node = parents[node]
        path.append(node)
        return path[::-1]
