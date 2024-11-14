import pytest

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
    return Simulation(graph, TOTAL_TIME, NUM_PATHS)

# Test that the graph is initialized correctly
def test_graph_init(graph):
    assert len(graph.nodes) == NUM_NODES
    assert len(graph.edges) == NUM_EDGES

# Test that the simulation is initialized correctly
def test_simulation_init(simulation):
    assert simulation.total_time == TOTAL_TIME
    assert len(simulation.roads) == len(simulation.graph.nodes)
    assert len(simulation.cars) == NUM_PATHS