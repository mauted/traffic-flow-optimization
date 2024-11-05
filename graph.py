from __future__ import annotations 
from collections import deque
import random
from util import partition_list
import networkx as nx
import matplotlib.pyplot as plt

random.seed(0)

class TrafficLightAgent:

    def __init__(self, edges: list[Edge], schedule=None):
        
        """
        Initializes the traffic light agent with a series of edges, and a schedule, where
        an schedule looks like:
        [(list[Edge], time), (list[Edge], time), (list[Edge], time), ..., (list[Edge], time)]
        and represents sequentially the subset of edges that are active, and the amount of time ticks it's active for. 
        """
        self.edges = edges
        if not schedule:
            self.schedule = self.generate_random_schedule()
        else:
            self.schedule = schedule
            
        # storing some constants that make it easier to make calculations
        self._period = sum(ticks for _, ticks in self.schedule)
    
        
    def generate_random_schedule(self, min_time: int = 10, max_time: int = 60):
        """Randomize a series of schedules for the edges given minimum and maximum time of a traffic light condition"""
        edges_copy = self.edges.copy()
        random.shuffle(edges_copy)
        shuffled = edges_copy
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

    def __init__(self, route: list[RoadSegment]):
        """
        - route: the route that the vehicle will take in the road network
        """
        self.id = Vehicle.__COUNTER
        Vehicle.__COUNTER += 1
        self.pos = 0
        self.route = route
        self.timer = route[self.pos].duration
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

    def clear_vehicles(self):
        self.vehicles = []
    
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
        in_roads = [RoadSegment() for _ in template.incoming]
        out_roads = [RoadSegment() for _ in template.outgoing]
        edges = [Edge(in_road, out_road) for in_road in in_roads for out_road in out_roads]

        return in_roads, out_roads, edges
    
    # TODO: For time ticks, something like this
    def tick(self, time):
        self.agent.tick(time) # this would activate and deactivate some of the edges