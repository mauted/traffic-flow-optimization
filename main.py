from old.simulation import RoadNetwork

def main(num_nodes: int, num_edges: int, num_vehicles: int):
    # this generates a graph with all nodes and edges, a vehicle system, and intersections with traffic light agents
    network = RoadNetwork(num_nodes, num_edges, num_vehicles)
    network.reset_cars()

    