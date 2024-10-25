from graph import RoadNetwork, Road, Edge
import random

random.seed(24)

class Car:
    
    __COUNTER = 0

    def __init__(self):
        self.id = Car.__COUNTER
        Car.__COUNTER += 1

class Traffic:

    def __init__(self, nodes: list[Road], edges: list[Edge], cars: list[Car]):
        self.nodes = nodes
        self.edges = edges 
        self.cars = cars

    def generate_start_end(self, num_paths: int) -> set[tuple[Road, Road]]: 
        """Generates a set of start and end nodes, representing a series of desired starting and ending points for the vehicles in the network."""
        paths = set()
        while len(paths) < num_paths:
            start, end = random.sample(self.nodes, 2)
            if (start, end) not in paths:
                paths.add((start, end))
        return paths
    
    def find_path(self, paths: set[tuple[Road, Road]]) -> list[list[Road]]:
        """
        Finds a path with A* search through the traffic for each path in paths, and returns the list of roads representing the route that the car takes. 
        Note that the path is not necessarily optimal, nor does it need to be. 
        """
        
    