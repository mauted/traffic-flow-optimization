from __future__ import annotations
import glob
import os
from graph import Graph, Node, generate_random_path, Edge
from draw_graph import draw_graph_from_list
import matplotlib.pyplot as plt
import networkx as nx
import random
from util import create_gif_from_images

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
    
    def __init__(self, node: Node, period: int = 60, schedule = None):
        self.edges = self.get_edges(node)
        self.period = period
        if schedule == None:
            self.schedule = self.generate_random_schedule()
        else:
            self.schedule = schedule
        self.active_edges = self.schedule[0][1]
        
    def get_edges(self, node: Node) -> list[Edge]:
        edges = []
        for from_node in node.incoming:
            edges.append(Edge(from_node, node))
        for to_node in node.outgoing:
            edges.append(Edge(node, to_node))
        return edges 

    def generate_random_schedule(self, min_duration: int = 5, max_duration: int = 15):
        elapsed_time = 0
        out = []
        while elapsed_time < self.period:
            duration = random.randint(min_duration, max_duration)
            elapsed_time += duration
            elapsed_time = min(elapsed_time, self.period)
            edge_subset = random.sample(self.edges, random.randint(1, len(self.edges)))
            out.append((elapsed_time, edge_subset))
        return out

    def tick(self, time: int):
        """
        Update the active edges based on the current simulation time.
        `time` is the global time of the simulation.
        """
        mod_time = time % self.period
        pos = 0
        while pos < len(self.schedule) and mod_time >= self.schedule[pos][0]:
            pos += 1
        self.active_edges = self.schedule[pos][1]
    

class Simulation:
    
    def __init__(self, graph: Graph, roads: list[Road], corr: dict[Node, Road], cars: list[LightningMcQueen], agents: list[TrafficLight], max_time: int = 1200):
        
        self.MAX_TIME = max_time
        self.INITIAL_NUM_CARS = len(cars)
        self.graph = graph
        self.roads = roads 
        self.corr = corr
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
            # if Edge(curr_seg.node.id, next_seg.node.id) in agent.active_edges:
            #     return True
            for active_edge in agent.active_edges:
                if curr_seg.node == active_edge.start and next_seg.node == active_edge.end:
                    return True
        return False
    
    def tick(self): 
        
        self.time += 1
        
        for light in self.agents:
            light.tick(self.time)
            
        for car in self.cars:
            
            curr_seg = car.path[car.pos]
            next_seg = car.path[car.pos + 1]

            if car.timer == 0:
                if self.is_active_edge(curr_seg, next_seg):
                    if len(next_seg.vehicles) < next_seg.capacity:
                        curr_seg.remove_mcqueen(car)
                        next_seg.add_mcqueen(car)
                        car.pos += 1
                        car.timer = car.path[car.pos].time

                if car.pos == len(car.path) - 1:
                    self.remove_vehicle(car)
                    continue

            else:
                car.timer -= 1
                
    
    def remove_vehicle(self, car: LightningMcQueen):
        # remove car from the simulation 
        self.cars.remove(car)
        # remove the car from the node it is on
        where = car.where()
        where.remove_mcqueen(car)

    
    def draw(self):
        # Create the graph structure
        G = nx.DiGraph()
        for node, neighbors in self.graph.adj_list.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        
        # Set the layout once and reuse it
        if not hasattr(self, "pos"):
            self.pos = nx.spring_layout(G, seed=0)  # Only compute layout once with a preset seed
        
        # Generate labels and colors based on road vehicles count
        labels = dict(zip([road.node.id for road in self.roads], [len(road.vehicles) for road in self.roads]))
        node_labels = {node: f"{labels[node.id]}" for node in G.nodes}

        node_colors = [plt.cm.RdYlGn(1 - labels[node.id] / self.corr[node].capacity) for node in G.nodes]
        
        # Draw the graph with the consistent layout
        plt.figure(figsize=(8, 6))
        nx.draw(G, self.pos, with_labels=False, node_color=node_colors, node_size=500, font_size=10, arrows=True)
        nx.draw_networkx_labels(G, self.pos, labels=node_labels, font_size=10, font_weight="bold", font_color="black")
        
        # Add text in the top-left corner with time step and cars remaining
        cars_remaining = sum(labels.values())
        plt.text(
            0.01, 0.99,  # Relative coordinates for text (top-left)
            f"Time Step: {self.time}\nCars Remaining: {cars_remaining}",
            transform=plt.gca().transAxes,  # Transform relative to axes
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
        )
        
        # Title and save the plot
        plt.title(f"Simulation Time: {self.time}")
        plt.savefig(f"graphs/simulation_time_{self.time}.png")


            

class LightningMcQueen:
    
    __COUNTER = 0
    
    def __init__(self, path: list[Road]):
        
        self.id = LightningMcQueen.__COUNTER
        LightningMcQueen.__COUNTER += 1
        self.path = path # set the car path
        self.pos = 0 # initialize car to the first road in the list of roads
        self.timer = self.path[self.pos].time # initialize the amount of time a car will remain on the nodes
        
    def where(self) -> Road:
        return self.path[self.pos]
    
    def has_completed_path(self):
        return self.pos == len(self.path) - 1

        
if __name__ == "__main__":

    random.seed(0)

    TOTAL_TIME = 100
    
    NUM_NODES = 5
    NUM_EDGES = 10
    NUM_PATHS = 50
    
    graph = Graph(NUM_NODES, NUM_EDGES)
    roads = [Road(node, capacity=random.randint(10, 20), time=random.randint(5, 10)) for node in graph.nodes]
    corr = dict(zip(graph.nodes, roads))
    cars = [LightningMcQueen(generate_random_path(roads, corr)) for _ in range(NUM_PATHS)]
    
    lights = [TrafficLight(node) for node in graph.nodes]
        
    sim = Simulation(graph=graph, 
                     roads=roads, 
                     corr=corr, 
                     cars=cars, 
                     agents=lights, 
                     max_time=TOTAL_TIME)

    # Remove all files in the graphs directory
    files = glob.glob('graphs/*')
    for f in files:
        os.remove(f)

    while sim.time < sim.MAX_TIME:
        sim.tick()
        if sim.cars:
            sim.draw()

    create_gif_from_images("graphs", "output.gif", duration=100)

    