from __future__ import annotations
from graph import Graph, Node, generate_random_path
from itertools import accumulate 
import random

class Road:
    
    def __init__(self, node: Node, capacity: int, time: int):
        
        # adding a correspondence between the nodes and the roads 
        self.node = node
        self.id = node.id
        self.capacity = capacity 
        self.time = time
        self.vehicles = []
    
    def add_mcqueen(self, car: LightningMcQueen):
        self.vehicles.append(car)

    def remove_mcqueen(self, car: LightningMcQueen):
        self.vehicles.remove(car)

class TrafficLight:
    
    def __init__(self, node: Node, total_time: int, schedule = None):
        self.edges = node.incoming + node.outgoing
        self.total_time = total_time
        if schedule == None:
            self.schedule = self.generate_random_schedule()
        else:
            self.schedule = schedule

        self.schedule_pos = 0

    def generate_random_schedule(self, min_time: int = 10, max_time: int = 60):
        elapsed_time = 0
        out = []
        while elapsed_time < self.total_time:
            duration = random.randint(min_time, max_time)
            elapsed_time += duration
            edge_subset = random.sample(self.edges, random.randint(1, len(self.edges)))
            out.append((elapsed_time, edge_subset))
        return out

    def tick(self, time: int):
        if self.schedule_pos >= len(self.schedule):
            return
        elapsed_time, active_edges = self.schedule[self.schedule_pos]
        if time >= elapsed_time:
            self.schedule_pos += 1
        self.active_edges = active_edges
    
    # def update_active_edges(self, time: int):
    #     relative_time = time % self._period
    #     for end_time, partition in self.schedule:
    #         if relative_time - end_time < 0: 
    #             self.active_edges = partition
    #             return 
    #     raise Exception("Active edges not found for this traffic agent, check that schedule is correct.")

class Simulation:
    
    def __init__(self, graph: Graph, cars: list[LightningMcQueen], agents: list[TrafficLight], total_time: int = 1200):
        
        self.total_time = total_time
        self.graph = graph
        self.cars = cars
        self.agents = agents
        
        # set variables to be the start of the simulation
        self.time = 0
    
    def tick(self): 

        if self.time >= self.total_time:
            return
        
        self.time += 1
        
        for light in self.agents:
            light.tick(self.time)
            
        for car in self.cars:
            
            curr_seg = car.route[car.pos]
            next_seg = car.route[car.pos + 1]

            if car.timer == 0:
                # TODO Implement is_active_edge
                if self.is_active_edge(curr_seg, next_seg):
                    if len(next_seg.vehicles) < next_seg.capacity:
                        curr_seg.remove_vehicle(car)
                        next_seg.add_vehicle(car)
                        car.pos += 1
                        car.timer = car.route[car.pos].duration

                if car.pos == len(car.route) - 1:
                    # TODO: FIGURE OUT WHAT TO DO WHEN A CAR FINISHES ITS ROUTE
                    self.remove_vehicle(car)
                    continue

            else:
                car.timer -= 1

class LightningMcQueen:
    
    __COUNTER = 0
    
    def __init__(self, path: list[Road]):
        
        self.id = LightningMcQueen.__COUNTER
        LightningMcQueen.__COUNTER += 1
        self.path = path # set the car path
        self.pos = 0 # initialize car to the first road in the list of roads
        self.timer = self.path[self.pos].time # initialize the amount of time a car will remain on the nodes
    
        
if __name__ == "__main__":

    TOTAL_TIME = 1200
    
    NUM_NODES = 10
    NUM_EDGES = 20
    NUM_PATHS = 10
    
    graph = Graph(NUM_NODES, NUM_EDGES)
    roads = [Road(node, random.randint(10, 20), random.randint(10, 20)) for node in graph.nodes]
    corr = dict(zip(graph.nodes, roads))
    cars = [LightningMcQueen(generate_random_path(roads, corr)) for _ in range(NUM_PATHS)]
    
    lights = [TrafficLight(node, total_time=TOTAL_TIME) for node in graph.nodes]
    
    sim = Simulation(graph, cars, lights, TOTAL_TIME)
    
    breakpoint()