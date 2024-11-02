from __future__ import annotations 
from collections import deque
import random
from util import partition_list
import networkx as nx
from networkx import check_planarity

random.seed(0)

class TrafficLightAgent:

    def __init__(self, edges: list[Edge], schedule: None):
        
        """
        Initializes the traffic light agent with a series of edges, and a schedule, where
        an schedule looks like:
        [(list[Edge], time), (list[Edge], time), (list[Edge], time), ..., (list[Edge], time)]
        and represents sequentially the subset of edges that are active, and the amount of time ticks it's active for. 
        """
        self.edges = edges
        if not schedule:
            self.generate_random_schedule()
        else:
            self.schedule = schedule
            
        # storing some constants that make it easier to make calculations
        self._period = sum(ticks for _, ticks in self.schedule)
        
        # setting the current active edges, this should speed up time needed to update a traffic light at the cost of memory
        self.active_edges = set(self.schedule[0][0])
        
    def generate_random_schedule(self, min_time: int = 10, max_time: int = 60):
        """Randomize a series of schedules for the edges given minimum and maximum time of a traffic light condition"""
        shuffled = random.shuffle(self.edges.copy())
        return [(partition, random.randint(min_time, max_time)) for partition in partition_list(shuffled)]    
    
    def find_active_edges(self, time: int) -> list[Edge]:
        """Updates the active edges of this traffic light agent based on the given global tick"""
        relative_time = time % self._period
        for edges, interval in self.schedule:
            if relative_time - interval < 0: 
                return edges 
            else:
                self.schedule -= interval
        raise Exception("Active edges not found for this traffic agent, check that schedule is correct.")
    
    def tick(self, time: int):
        """Changes the state of this traffic light agent and the edges it controls depending on the time"""
        
        next_actives = set(self.find_active_edges(time))
        for edge in self.active_edges:
            edge.active = False
        for edge in self.next_actives:
            edge.active = True
        self.active_edges = next_actives


class Vehicle:
    
    __COUNTER = 0

    def __init__(self, route: list[RoadSegment], road_net: RoadNetwork):
        """
        - route: the route that the vehicle will take in the road network
        """
        self.id = Vehicle.__COUNTER
        Vehicle.__COUNTER += 1
        self.pos = 0
        self.route = route
        self.timer = route[self.pos].duration
        self.road_net = road_net
        self.route_complete = False        

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

    def add_vehicle(self, vehicle: Vehicle):
        if len(self.vehicles) < self.capacity:
            self.vehicles.append(vehicle)
        else:
            raise Exception("Road segment is at capacity")
        
    def remove_vehicle(self, vehicle: Vehicle):
        self.vehicles.remove(vehicle)
    
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
        # NOTE: This behavior should change depending on the starting condition of the traffic
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

        for node in nodes:
            if node in in_roads:
                in_roads.append(nodes[node])
            elif node in out_roads:
                out_roads.append(nodes[node])
            else:
                in_roads.append(nodes[node])
                out_roads.append(nodes[node]) 

        # constructing the connections
        edges = []
        for in_road in in_roads:
            for out_road in out_roads:
                in_road.add_outgoing(out_road)
                out_road.add_incoming(in_road)
                edges.append(Edge(in_road, out_road))

        return in_roads, out_roads, edges
    
    # TODO: For time ticks, something like this
    def tick(self, time):
        self.agent.tick(time) # this would activate and deactivate some of the edges

class RoadNetwork:
    """
    The domain of the environment for the RL simulations.
    """

    def __init__(self, nodes: list[RoadSegment], edges: list[Edge], intersections: list[Intersection], vehicles: list[Vehicle]):
        self.nodes = nodes
        self.edges = edges 
        self.vehicles = vehicles
        self.time = 0
        self.completed_route_counter = 0
        
        
    def tick(self):
        """Ticks a time in this simulation of the road network"""
        self.time += 1
        # TODO: This I imagine is how we make the simulation "go", sort of like 'on_tick' for big bang in Racket, the idea that a tick that happens in the big simulation class (maybe that is this RoadNetwork class) would propagate the tick throughout the items in that road network. 
        # change the traffic light conditions for this tick
        # something like intersection.tick() for intersection in self.intersections
        # something for vehicles to move 

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



    def add_vehicle(self, vehicle: Vehicle):
        """
        Adds a vehicle to the road network
        """
        self.vehicles.append(vehicle)

    def remove_vehicle(self, vehicle: Vehicle):
        """
        Removes a vehicle from the road network
        """
        self.vehicles.remove(vehicle)

    def is_active_edge(self, start: RoadSegment, end: RoadSegment) -> bool:
        """
        Returns whether the edge between start and end is active
        """
        for edge in self.edges:
            if edge.start == start and edge.end == end:
                return edge.active
        return False

    @staticmethod
    def _build_supergraph(num_nodes: int, is_planar: bool = False) -> list[PreIntersection]:
        """
        Constructs a randomly generated connected directed graph with num_nodes pre-intersection nodes.
        Each node is connected to multiple other nodes.
        """
        if num_nodes < 2:
            raise ValueError("Number of nodes must be at least 2")

        # Initialize nodes
        pre_nodes = [PreIntersection() for _ in range(num_nodes)]
        possible_nodes = pre_nodes[:]

        # Connect every node with multiple others
        for start in pre_nodes:
            # Set a target number of edges each node will have
            target_outgoing = random.randint(2, num_nodes - 1)
            possible_nodes.remove(start)
            
            # Create outgoing edges to reach the target connectivity
            for end in random.sample(possible_nodes, target_outgoing):
                if end not in start.outgoing:
                    start.add_outgoing(end)
                    end.add_incoming(start)

            possible_nodes.append(start)

        # After every node has outward connections, add edges to nodes without incoming connections
        for end in pre_nodes:
            if not end.incoming:
                target_outgoing = random.randint(2, num_nodes - 1)
                possible_nodes.remove(end)
                for start in random.sample(possible_nodes, target_outgoing):
                    start.add_outgoing(end)
                    end.add_incoming(start)
                possible_nodes.append(end)

        return pre_nodes
    
    @staticmethod
    def build_complete_network(num_nodes: int, is_planar: bool = False) -> RoadNetwork:
        """
        Generates the complete road network, from a given number of edges.
        """
        pre_intersections = RoadNetwork._build_supergraph(num_nodes, is_planar)
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
        
        return RoadNetwork(nodes=nodes, edges=edges, intersections=intersections, vehicles=[])
        
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
                return self.traceback(node, parents)
            else: 
                explored.add(node)
                for succ in node.outgoing:
                    if succ not in explored and succ not in frontier:
                        parents[succ] = node
                        frontier.append(succ)
        
        return None
    
    def traceback(self, node: RoadSegment, parents: dict[RoadSegment, RoadSegment]):
        """Traces back the path followed by the node in the parents dictionary"""
        path = []
        while node in parents: 
            path.append(node)
            node = parents[node]
        path.append(node)
        return path[::-1]
