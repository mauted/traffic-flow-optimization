from simulation import Simulation, TrafficLight, LightningMcQueen, Road
from environment import TrafficEnvironment
from graph import Edge, Graph, generate_random_path
from itertools import product
import random 
import gym


def adjacent_increments(time: int, step: int, max: int):
    """Calculates the adjacent increments to the time, given a maximum time."""
    adjs = []
    if time - step >= 0:
        adjs.append(time - step)
    if time + step <= max:
        adjs.append(time + step)
    adjs.append(time)
    return adjs

# TODO: This function needs to take in the same parameters, but just return a list of 4 tuples, which represent the list of schedules possible in this simulation for this agent in the next time step.
def get_adjacent_schedules(sim: Simulation, agent: TrafficLight, step: int): 
    """
    Generates the adjacent schedules for the single agent in the simulation. 
    This also sets the hard limit that traffic light times can only be increased/decreased in discrete step sizes.
    The hard limit here is not set for the whole simulation as we would like those to remain dynamically adjustable.
    This is unfortunately quite expensive to compute, but this should limit the action space to a feasible amount for Q-learning. 
    """
    possible_times = [adjacent_increments(time, step, sim.MAX_LIGHT_TIME) for time, _ in agent.schedule]
    # this cartesian product represents all the possible neighboring schedules for this particular schedule 
    cartesian_prod = list(product(*possible_times))
    # this can probably be a list comprehension, but would that be comprehensible? probably no :)
    possible_schedules = [] 
    for times in cartesian_prod:
        current_schedule = [] 
        for i, time in enumerate(times):
            current_schedule.append((time, agent.schedule[i][1])) # time and the edges it corresponds to
        possible_schedules.append(current_schedule)
    return possible_schedules


# TODO: We shouldn't need this function after refactoring
def make_hashable(data: dict[int, list[tuple[int, set[Edge]]]]) -> dict[int, tuple[tuple[int, frozenset[int]]]]:
    """Turns the schedule into a hashable type"""
    """
    Recursively converts the schedule structure into a hashable type.
    Replaces:
      - list -> tuple
      - set -> frozenset
    """
    if isinstance(data, list):
        return tuple(make_hashable(item) for item in data)
    elif isinstance(data, set):
        return frozenset(make_hashable(item) for item in data)
    elif isinstance(data, dict):
        return tuple((key, make_hashable(value)) for key, value in data.items())
    elif isinstance(data, tuple):
        return tuple(make_hashable(item) for item in data)
    elif isinstance(data, Edge):
        return (Edge.start.id, Edge.end.id)
    else:
        return schedule
        
# TODO: (luisa probably) needs to add a checkpointing system back into this method

def q_learning(env: gym.Env, num_episodes, gamma=0.9, epsilon=0.9):

    Q = {} 
    num_updates = {}
    
    # for every episode        
    for _ in range(num_episodes):
        
        # reset the environment 
        observation = env.reset()
        terminated = False
        agent_index = 0
        
        # while not an ending state
        while not terminated:
                        
            prob = random.uniform(0, 1)
            
            # find the neighbors of this current state 
            agent = env.sim.agents[agent_index]
            
            adj_schedules = get_adjacent_schedules(env.sim, agent, 5) # NOTE: This is hardcoding this increment to steps of 5
            
            # TODO: Should not need this anymore since observation should be a 2 tuple
            hashable_observation = make_hashable(observation)
            
            # select action 
            if prob < epsilon:
                action = random.choice(adj_schedules)
            else:
                # TODO: This should iterate over all possible actions and choose the best one according to the Q table
                obs = ((action, value) for (obs, action), value in Q.items() if obs == hashable_observation)
                action = max(obs, key=lambda item: item[1])[0]
            
            # TODO: Should not need this anymore                
            hashable_action = make_hashable(action)
                        
            # get the new observation                    
            new_observation, reward, terminated = env.step({agent.id: action})
            
            # TODO: Should not need this anymore                
            hashable_new_observation = make_hashable(new_observation)
            
            # TODO: This calculation should be exactly like the reinforcement learning homework that we did, except that the Q table should be a dictionary that takes a (observation, action) and returns the result. The observation is a 2 tuple and the action is a #num-partitions tuple
            # calculating the Q values in parts 
            alpha = 1 / (1 + num_updates.get((hashable_observation, hashable_action), 0))
            items = [value for (obs, _), value in Q.items() if obs == hashable_new_observation]  
            if len(items) == 0:
                gamma_term = 0
            else:
                gamma_term = gamma * max(items) 
            Q[(hashable_observation, hashable_action)] = Q.get((hashable_observation, hashable_action), 0) + (alpha * (reward + gamma_term - Q.get((hashable_observation, hashable_action), 0)))
 
            # updating the num_updates matrix 
            num_updates[hashable_observation, hashable_action] = num_updates.get((hashable_observation, hashable_action), 0) + 1 
            # updating observation
            observation = new_observation
            
            agent_index = (agent_index + 1) % len(env.sim.agents)
                    
        # update epsilon at the end of the episode 
        epsilon = 0.9999 * epsilon 
        
    return Q

if __name__ == "__main__":
    
    random.seed(0)

    TOTAL_TIME = 100
    
    NUM_NODES = 3
    NUM_EDGES = 5
    NUM_PATHS = 3
    NUM_PARTITIONS = 4
    MAX_LIGHT_TIME = 30
    TIME_RATIO = 3
    
    graph = Graph(NUM_NODES, NUM_EDGES)
    roads = [Road(node, capacity=random.randint(10, 20), time=random.randint(5, 10)) for node in graph.nodes]
    corr = dict(zip(graph.nodes, roads))
    cars = [LightningMcQueen(generate_random_path(roads, corr)) for _ in range(NUM_PATHS)]
        
    lights = [TrafficLight(node, NUM_PARTITIONS) for node in graph.nodes] 
        
    sim = Simulation(graph=graph, 
                     roads=roads, 
                     corr=corr, 
                     cars=cars, 
                     agents=lights, 
                     max_time=TOTAL_TIME,
                     num_partitions=NUM_PARTITIONS,
                     max_light_time=MAX_LIGHT_TIME)
    
    env = TrafficEnvironment(sim, TIME_RATIO)
    q_learning(env, 1, 3)