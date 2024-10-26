from graph import RoadNetwork, Road, Edge, Car
from collections import deque
import random

random.seed(24)

class Traffic:

    def __init__(self, nodes: list[Road], edges: list[Edge], cars: list[Car]):
        self.nodes = nodes
        self.edges = edges 
        self.cars = cars
        
    def generate_paths(self, num: int) -> list[list[Road]]:
        """
        Generates all paths for the vehicles of this traffic network
        """
        paths = []
        start_ends = self.generate_start_end(num)
        for start, end in start_ends:
            paths.append(self.dfs(start, end))
        return paths

    def generate_start_end(self, num_paths: int) -> set[tuple[Road, Road]]: 
        """Generates a set of start and end nodes, representing a series of desired starting and ending points for the vehicles in the network."""
        paths = set()
        while len(paths) < num_paths:
            start, end = random.sample(self.nodes, 2)
            if (start, end) not in paths:
                paths.add((start, end))
        return paths
    
    def dfs(self, start: Road, end: Road) -> list[Road]:
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
    
    def _traceback(self, node: Road, parents: dict[Road, Road]):
        """Traces back the path followed by the node in the parents dictionary"""
        path = []
        while node in parents: 
            path.append(node)
            node = parents[node]
        path.append(node)
        return path[::-1]
    
                

            
    


    