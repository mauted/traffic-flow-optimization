from simulation import Simulation, TrafficLight, LightningMcQueen, Road
from environment import TrafficEnvironment
from graph import Edge, Graph, generate_random_path
from itertools import product
import numpy as np
import random 
import gym


def calculate_nS(sim: Simulation):
    """
    Calculates the number of possible observations/states in the observation/state space
    This is defined in the multi agent case as: 
    # of agents * maximum # of vehicles in the simulation * 2
    """
    return len(sim.agents) * sim.INITIAL_NUM_CARS * 2


def calculate_nA(sim: Simulation):
    """
    Calculates the number of possible actions in the action space. 
    This is defined in the multi-agent case as: 
    # of agents * # partitions * maximum light time in the simulation
    """
    return len(sim.agents) * sim.NUM_PARTITIONS * sim.MAX_LIGHT_TIME


def adjacent_increments(time: int, step: int, max: int):
    """Calculates the adjacent increments to the time, given a maximum time."""
    adjs = []
    if time - step >= 0:
        adjs.append(time - step)
    if time + step <= max:
        adjs.append(time + step)
    return adjs

def get_adjacent_schedules(sim: Simulation, step: int): 
    """
    Generates the adjacent schedules for each agent in the simulation. 
    This also sets the hard limit that traffic light times can only be increased/decreased in discrete step sizes.
    The hard limit here is not set for the whole simulation as we would like those to remain dynamically adjustable.
    This is unfortunately quite expensive to compute, but this should limit the action space to a feasible amount for Q-learning. 
    """
    adj_schedules = {}
    for agent in sim.agents:
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
        adj_schedules[agent] = possible_schedules
    # now take the cartesian product again
    cartesian_product_agent = [dict(zip(adj_schedules.keys(), combination)) for combination in product(*adj_schedules.values())]
    return cartesian_product_agent


def make_hashable(schedule: dict[int, list[tuple[int, set[Edge]]]]) -> dict[int, tuple[tuple[int, frozenset[int]]]]:
    """Turns the schedule into a hashable type"""
    """
    Recursively converts the schedule structure into a hashable type.
    Replaces:
      - list -> tuple
      - set -> frozenset
    """
    if isinstance(schedule, list):
        return tuple(make_hashable(item) for item in schedule)
    elif isinstance(schedule, set):
        return frozenset(make_hashable(item) for item in schedule)
    elif isinstance(schedule, dict):
        return tuple((key, make_hashable(value)) for key, value in schedule.items())
    elif isinstance(schedule, tuple):
        return tuple(make_hashable(item) for item in data)
    else:
        return data
        

def q_learning(env: gym.Env, num_episodes, checkpoints, gamma=0.9, epsilon=0.9):
    """
    Q-learning algorithm.

    Parameters:
    - num_episodes (int): Number of Q-value episodes to perform.
    - checkpoints (list): List of episode numbers at which to record the optimal value function..

    Returns:
    - Q (numpy array): Q-values of shape (nS, nA) after all episodes.
    - optimal_policy (numpy array): Optimal policy, np array of shape (nS,), ordered by state index.
    - V_opt_checkpoint_values (list of numpy arrays): A list of optimal value function arrays at specified episode numbers.
      The saved values at each checkpoint should be of shape (nS,).
    """

    
    Q = {} 
    num_updates = {}
    checkpoints = []
    
    # for every episode        
    for episode in range(num_episodes):
        
        # reset the environment 
        observation = env.reset()
        terminated = False
        
        # while not an ending state
        while not terminated:
            
            prob = random.uniform(0, 1)
            
            # find the neighbors of this current state 
            adj_schedules = get_adjacent_schedules(env.sim, 5) # NOTE: This is hardcoding this increment to steps of 5
                        
            # select action 
            if prob < epsilon:
                action = random.choice(adj_schedules)
            else:
                action_reward = dict(zip(adj_schedules, [Q.get((observation, action), 0) for action in adj_schedules]))
                action = max(action_reward, key=action_reward.get)
                        
            # get the new observation                    
            new_observation, reward, terminated = env.step(action)
                        
            # calculating the Q values in parts 
            alpha = 1 / (1 + num_updates.get((observation, action), 0))
            gamma_term = gamma * max(Q[new_observation, :])
            Q[(observation, action)] += alpha * (reward + gamma_term - 
                                             Q.get((observation, action), 0))
 
            # updating the num_updates matrix 
            num_updates[observation, action] = num_updates.get((observation, action), 0) + 1 
            # updating observation
            observation = new_observation
                    
        # update epsilon at the end of the episode 
        epsilon = 0.9999 * epsilon 
        
        # if the episode is a checkpoint episode, checkpoint Q
        if episode + 1 in checkpoints:
            checkpoints.append(np.max(Q, axis=1))
        
        # set the optimal policy 
        optimal_policy = np.argmax(Q, axis=1)
    
    return Q, optimal_policy, checkpoints

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