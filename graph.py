from __future__ import annotations 
from enum import Enum

class RoadSegmentType(Enum):
    IN = 0,
    OUT = 1,
    BOTH = 2

class RoadNetwork:
    
    def __init__(self, roads, edges):
        # subject to change but I assume this road network will have a list of nodes
        self.roads = roads
        self.edges = edges 
    
    def generate_supergraph(self, num_nodes: int):
        self.V = []
        self.E = []
        for _ in range(num_nodes):
            self.V.append(Road())
        # Add edges to the graph that ensure that the graph is connected
        

        



class Road:
    
    __COUNTER = 0
    
    def __init__(self, outgoing: list[Road] = [], incoming: list[Road] = [], duration: int = 30, ):
        self.id = Road.__COUNTER
        Road.__COUNTER += 1
        self.outgoing = outgoing
        self.incoming = incoming
        self.cars = []
    
    def add_incoming(self, road):
        self.incoming.append(road)
    
    def add_outgoing(self, road):
        self.outgoing.append(road)
    
    def move_car(self, car: Car):
        pass
    
class Edge:
    
    __COUNTER = 0

    def __init__(self, start: Road, end: Road):
        self.id = Edge.__COUNTER
        Edge.__COUNTER += 1 
        self.id = Edge.__COUNTER
        Edge.__COUNTER += 1
        self.start = start
        self.end = end
        self.start.outgoing.append(self.end)
        self.end.incoming.append(self.start)


class Intersection:

    __COUNTER = 0

    def __init__(self, in_roads: list[Road], out_roads: list[Road], edges: list[Edge]): 
        self.id = Intersection.__COUNTER
        Intersection.__COUNTER += 1
        self.in_roads = in_roads
        self.out_roads = out_roads
        self.edges = edges

    def construct_intersection(roads: list[RoadSegmentType]) -> Intersection:
        """
        Takes an input of the form [IN, OUT, BOTH,...] and returns a constructed intersection
        with the specified number of roads. Each road going into the intersection connects to each 
        road going out of the intersection.
        """

        # constructing the nodes to the graph
        in_roads, out_roads = [], []
        for road in roads: 
            if road == RoadSegmentType.IN:
                in_roads.append(Road())            
            elif road == RoadSegmentType.OUT:
                out_roads.append(Road())
            else:
                out_roads.append(Road())
                in_roads.append(Road())

        # constructing the connections
        edges = []
        for in_road in in_roads:
            for out_road in out_roads:
                in_road.add_outgoing(out_road)
                out_road.add_incoming(in_road)
                edges.append(Edge(in_road, out_road))

        return Intersection(in_roads, out_roads, edges)
        

if __name__ == "__main__":
    G = GraphGenerator(10)