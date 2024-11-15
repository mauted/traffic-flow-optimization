import random
from graph import generate_random_path
import pytest
from simulation import LightningMcQueen, Road, TrafficLight

# Constants
TOTAL_TIME = 1200
NUM_NODES = 10
NUM_EDGES = 20
NUM_PATHS = 10

# Make a graph object that resets the graph after each test
@pytest.fixture
def graph():
    from simulation import Graph
    return Graph(NUM_NODES, NUM_EDGES)

# Make a simulation object that resets the simulation after each test
@pytest.fixture
def simulation(graph):
    from simulation import Simulation
    roads = [Road(node, random.randint(10, 20), random.randint(1, 2)) for node in graph.nodes]
    corr = dict(zip(graph.nodes, roads))
    cars = [LightningMcQueen(generate_random_path(roads, corr)) for _ in range(NUM_PATHS)]
    
    lights = [TrafficLight(node) for node in graph.nodes]
        
    return Simulation(graph, roads, cars, lights, TOTAL_TIME)

# Test that the simulation is initialized correctly
def test_simulation_init(simulation):
    assert simulation.TOTAL_TIME == TOTAL_TIME
    assert len(simulation.roads) == len(simulation.graph.nodes)
    assert len(simulation.cars) == NUM_PATHS

