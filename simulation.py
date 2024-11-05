from graph import RoadSegment, Edge, Intersection, Vehicle, PreIntersection
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import deque

class RoadNetwork:
    """
    The domain of the environment for the RL traffic flow simulations.
    """

    def __init__(self, num_nodes: int, num_edges: int):
        """Initializes this road network by building a graph"""
        self.nodes, self.edges, self.intersections = self._build_complete_network(num_nodes, num_edges)


    def generate_vehicles(self, num_vehicles: int):
        """Generates a problem for this road network, which takes the form of a list of vehicles that each have a path to follow"""
        self.vehicles = self.generate_vehicles(num_vehicles)

    
    def tick(self):
        """Ticks a time in this simulation of the road network"""

        self.time += 1

        for intersection in self.intersections:
            intersection.tick(self.time)

        for vehicle in self.vehicles:

            curr_seg = vehicle.route[vehicle.pos]
            next_seg = vehicle.route[vehicle.pos + 1]

            if vehicle.timer == 0:
                if self.is_active_edge(curr_seg, next_seg):
                    if len(next_seg.vehicles) < next_seg.capacity:
                        curr_seg.remove_vehicle(vehicle)
                        next_seg.add_vehicle(vehicle)
                        vehicle.pos += 1
                        vehicle.timer = vehicle.route[vehicle.pos].duration

                if vehicle.pos == len(vehicle.route) - 1:
                    self.completed_route_counter += 1
                    self.remove_vehicle(vehicle)
                    continue

            else:
                vehicle.timer -= 1

    def reset_simulation(self):
        pass
    

    def save_network_figure(self):
        """
        Saves a figure of the road network
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from([(edge.start, edge.end) for edge in self.edges])

        pos = nx.spring_layout(G)

        plt.figure(figsize=(10, 8))
        
        # Calculate node colors based on the number of vehicles in each node
        node_colors = []
        for node in self.nodes:
            ratio = len(node.vehicles) / node.capacity if node.capacity > 0 else 0
            node_colors.append(ratio)
        
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=plt.cm.viridis)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

        edges = G.edges()
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=0.2')
        nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=20, connectionstyle='arc3,rad=-0.2')

        plt.title("Directed Graph T")
        plt.savefig(f"figs/road_network_at_time_{self.time:03d}.png")
        

    @staticmethod
    def _build_supergraph(num_nodes: int, num_edges: int) -> list[PreIntersection]:
        """
        Constructs a randomly generated connected directed graph with num_nodes pre-intersection nodes
        and num_edges edges.
        """
        if num_nodes < 2:
            raise ValueError("Number of nodes must be at least 2")
        if num_edges < num_nodes - 1:
            raise ValueError("Number of edges must be at least num_nodes - 1 to ensure connectivity")

        # Initialize nodes
        pre_nodes = [PreIntersection() for _ in range(num_nodes)]
        possible_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        random.shuffle(possible_edges)

        # Ensure the graph is connected by creating a spanning tree first
        connected_nodes = set()
        connected_nodes.add(0)
        while len(connected_nodes) < num_nodes:
            for i, j in possible_edges:
                if (i in connected_nodes) != (j in connected_nodes):
                    pre_nodes[i].add_outgoing(pre_nodes[j])
                    pre_nodes[j].add_incoming(pre_nodes[i])
                    connected_nodes.add(i)
                    connected_nodes.add(j)
                    possible_edges.remove((i, j))
                    break

        # Add remaining edges randomly until reaching num_edges
        current_edges = num_nodes - 1
        while current_edges < num_edges:
            i, j = possible_edges.pop()
            pre_nodes[i].add_outgoing(pre_nodes[j])
            pre_nodes[j].add_incoming(pre_nodes[i])
            current_edges += 1

        return pre_nodes
    
    @staticmethod
    def _build_complete_network(num_nodes: int, num_edges: int) -> tuple[list[RoadSegment], list[Edge], list[Intersection]]:
        """
        Generates the complete road network, from a given number of edges.
        """
        pre_intersections = RoadNetwork._build_supergraph(num_nodes, num_edges)
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


    def _generate_vehicles(num_vehicles: int) -> list[Vehicle]:
        """
        Generates all paths for the vehicles of this traffic network
        """
        vehicles = []
        endpoints = RoadNetwork._generate_vehicle_endpoints(num_vehicles)
        for start, end in endpoints:
            path = RoadNetwork._dfs(start, end)
            corvette = Vehicle(path)
            vehicles.append(corvette)
        return vehicles


    def _generate_vehicle_endpoints(self, num_paths: int) -> set[tuple[RoadSegment, RoadSegment]]: 
        """Generates a set of start and end nodes, representing a series of desired starting and ending points for the vehicles in the network."""
        paths = set()
        while len(paths) < num_paths:
            start, end = random.sample(self.nodes, 2)
            if (start, end) not in paths:
                paths.add((start, end))
        return paths
    

    @staticmethod
    def _find_paths(paths: set[tuple[RoadSegment, RoadSegment]]) -> list[list[RoadSegment]]:
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
                    return RoadNetwork.traceback(node, parents)
                else: 
                    explored.add(node)
                    for succ in node.outgoing:
                        if succ not in explored: 
                            frontier.append(succ)
            
            found_paths.append(None)
        return found_paths


    @staticmethod        
    def _dfs(start: RoadSegment, end: RoadSegment) -> list[RoadSegment]:
        """Finds the path from start to end within the road network and returns the path as a list of roads"""
    
        parents = {}
            
        frontier = deque()
        frontier.append(start)
        explored = set()
        
        while len(frontier) != 0: 
            node = frontier.pop()
            if node == end:
                return RoadNetwork._traceback(node, parents)
            else: 
                explored.add(node)
                for succ in node.outgoing:
                    if succ not in explored and succ not in frontier:
                        parents[succ] = node
                        frontier.append(succ)
        
        return None
    

    @staticmethod
    def _traceback(node: RoadSegment, parents: dict[RoadSegment, RoadSegment]):
        """Traces back the path followed by the node in the parents dictionary"""
        path = []
        while node in parents: 
            path.append(node)
            node = parents[node]
        path.append(node)
        return path[::-1]


    def _add_vehicle(self, vehicle: Vehicle):
        """
        Adds a vehicle to the road network
        """
        self.vehicles.append(vehicle)


    def _remove_vehicle(self, vehicle: Vehicle):
        """
        Removes a vehicle from the road network
        """
        self.vehicles.remove(vehicle)


