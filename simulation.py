from __future__ import annotations
from graph import Graph, Node, Edge
from util import partition_list
from itertools import accumulate 
import random

# hello!!!!

class TrafficLight:
    
    def __init__(self, node: Node, schedule: None):
        self.edges = node.incoming + node.outgoing
        if not schedule:
            self.generate_random_schedule()
        else:
            self.schedule = schedule
            
        # storing some constants that make it easier to make calculations
        self._period = sum(ticks for _, ticks in self.schedule)
        
        # setting the current active edges, this should speed up time needed to update a traffic light at the cost of memory
        self.active_edges = set(self.schedule[0][0])
        
    def generate_random_schedule(self, min_time: int = 10, max_time: int = 60):
        shuffled = random.shuffle(self.edges.copy())
        partition = partition_list(shuffled)
        times = [random.randint(min_time, max_time) for _ in partition]
        return [zip(list(accumulate(times)), partition)]
    
    def update_active_edges(self, time: int):
        relative_time = time % self._period
        for end_time, partition in self.schedule:
            if relative_time - end_time < 0: 
                self.active_edges = partition
                return 
        raise Exception("Active edges not found for this traffic agent, check that schedule is correct.")

class Simulation:
    
    def __init__(self, graph: Graph, cars: list[LightningMcQueen], agents: list[TrafficLight]):
        
        self.graph = graph
        self.cars = cars
        self.agents = agents
        
        # set variables to be the start of the simulation
        self.time = 0
    
    def tick(self): 
        
        self.time += 1
        
        for light in self.agents:
            light.tick(self.time)
            
        for car in self.cars:
            
            curr_seg = car.route[car.pos]
            next_seg = car.route[car.pos + 1]

            if car.timer == 0:
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
        self.timer = self.path[self.pos].time # initialize the amount of time a car will remain on the nodeß
    
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

    def remove_vehicle(self, car: LightningMcQueen):
        self.vehicles.remove(car)
        