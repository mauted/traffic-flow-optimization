from graph import Graph

class Car:
    
    __COUNTER = 0

    def __init__(self):
        self.id = Car.__COUNTER
        Car.__COUNTER += 1

class Traffic:

    def __init__(self, graph: Graph, cars: list[Car]):
        self.graph = graph 
        self.cars = cars

    