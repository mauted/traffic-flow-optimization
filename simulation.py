from __future__ import annotations
from graph import Graph, Node, generate_random_path
from draw_graph import draw_graph_from_list
import matplotlib.pyplot as plt
import networkx as nx
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
        elapsed_time, active_edges = self.schedule[self.schedule_pos]
        self.active_edges = active_edges
        if self.schedule_pos >= len(self.schedule):
            return
        if time >= elapsed_time:
            self.schedule_pos += 1
    

class Simulation:
    
    def __init__(self, graph: Graph, roads: list[Road], cars: list[LightningMcQueen], agents: list[TrafficLight], total_time: int = 1200):
        
        self.total_time = total_time
        self.graph = graph
        self.roads = roads 
        self.cars = cars
        self.agents = agents
        
        # set variables to be the start of the simulation
        self.time = 0
        
        # add mcqueens here 
        for car in cars:
            start = car.path[0]
            start.add_mcqueen(car)

    def is_active_edge(self, curr_seg: Road, next_seg: Road):
        for agent in self.agents:
            if curr_seg in agent.active_edges and next_seg in agent.active_edges:
                return True
        return False
    
    def tick(self): 

        if self.time >= self.total_time:
            return
        
        self.time += 1
        
        for light in self.agents:
            light.tick(self.time)
            
        for car in self.cars:
            
            curr_seg = car.path[car.pos]
            next_seg = car.path[car.pos + 1]

            if car.timer == 0:
                # TODO Implement is_active_edge
                if self.is_active_edge(curr_seg, next_seg):
                    if len(next_seg.vehicles) < next_seg.capacity:
                        curr_seg.remove_vehicle(car)
                        next_seg.add_vehicle(car)
                        car.pos += 1
                        car.timer = car.path[car.pos].duration

                if car.pos == len(car.route) - 1:
                    # TODO: FIGURE OUT WHAT TO DO WHEN A CAR FINISHES ITS ROUTE
                    self.remove_vehicle(car)
                    continue

            else:
                car.timer -= 1
                
    def draw(self):
                
        G = nx.DiGraph()
        for node, neighbors in self.graph.adj_list.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        pos = nx.spring_layout(G) 
        
        labels = dict(zip([road.node.id for road in self.roads], [len(road.vehicles) for road in self.roads]))
                
        node_labels = {node: f"{labels[node.id]}" for node in G.nodes}
        plt.figure(figsize=(8, 6))
        
        nx.draw(G, pos, with_labels=False, node_color="lightblue", node_size=500, font_size=10, arrows=True)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold", font_color="black")
        
        plt.show()
            

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
    
    NUM_NODES = 5
    NUM_EDGES = 10
    NUM_PATHS = 10
    
    graph = Graph(NUM_NODES, NUM_EDGES)
    roads = [Road(node, random.randint(10, 20), random.randint(10, 20)) for node in graph.nodes]
    corr = dict(zip(graph.nodes, roads))
    cars = [LightningMcQueen(generate_random_path(roads, corr)) for _ in range(NUM_PATHS)]
    
    lights = [TrafficLight(node, total_time=TOTAL_TIME) for node in graph.nodes]
        
    sim = Simulation(graph, roads, cars, lights, TOTAL_TIME)

    sim.draw()
    breakpoint()