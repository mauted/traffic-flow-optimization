import osmnx as ox
import matplotlib.pyplot as plt

def download_and_plot_road_network(place_name):
    # Download road network data for a specific location
    G = ox.graph_from_place(place_name, network_type='drive')

    # Print the information about the road network
    print(ox.basic_stats(G))

    # Plot the road network using osmnx
    fig, ax = ox.plot_graph(G, show=False, close=False)
    plt.title(f"Road Network of {place_name}")
    plt.savefig(f"figs/road_network_{place_name}.png")

if __name__ == "__main__":
    # Specify the location
    place_name = "Boston, USA"
    
    # Download and plot the road network
    download_and_plot_road_network(place_name)
