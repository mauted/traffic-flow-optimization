from simulation import Simulation, TrafficLight, LightningMcQueen, Road
from environment import TrafficEnvironment
from graph import Edge, Graph, generate_random_path
from itertools import product
import random 
import gym

"""Cursor no"""


def adjacent_increments(time: int, step: int, max: int):
    """Calculates the adjacent increments to the time, given a maximum time."""
    adjs = []
    if time - step >= 0:
        adjs.append(time - step)
    if time + step <= max:
        adjs.append(time + step)
    adjs.append(time)
    return adjs


def get_adjacent_schedules(sim: Simulation, agent: TrafficLight, step: int) -> list[tuple[int]]: 
    """
    Generates the adjacent schedules for the single agent in the simulation. 
    This also sets the hard limit that traffic light times can only be increased/decreased in discrete step sizes.
    The hard limit here is not set for the whole simulation as we would like those to remain dynamically adjustable.
    This is unfortunately quite expensive to compute, but this should limit the action space to a feasible amount for Q-learning. 
    """
    possible_times = [adjacent_increments(time, step, sim.MAX_LIGHT_TIME) for time, _ in agent.schedule]
    cartesian_prod = list(product(*possible_times))
    return cartesian_prod


def build_schedule_around(sim: Simulation, agent: TrafficLight, updated_sched: tuple[int]):
    """Builds the full schedule around the single agent's updated schedule"""
    return [a.schedule if a.id != agent.id else updated_sched for a in sim.agents]

        
# TODO: (luisa probably) needs to add a checkpointing system back into this method

def q_learning(env: gym.Env, num_episodes: int, time_increments: int = 5, gamma=0.9, epsilon=0.9):

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
            
            # get the agent whose schedule to change in this iteration
            agent = env.sim.agents[agent_index]
            # get all the adjacent schedules for this agent, so the next possible schedules to assign this agent.
            adj_schedules_agent = get_adjacent_schedules(env.sim, agent, time_increments)
            adj_schedules_all = [build_schedule_around(env.sim, agent, sched) for sched in adj_schedules_agent]
            
            # select action 
            if prob < epsilon:
                action = random.choice(adj_schedules_all)
            else:
                max_value = 0
                max_action_index = random.randint(0, len(adj_schedules_all) - 1)
                for i, a in enumerate(adj_schedules_all):
                    value = Q.get(observation, {}).get(action, 0)
                    if value > max_value:
                        max_action_index = i
                        max_value = value
                action = adj_schedules_all[max_action_index]
                
            # get the new observation                    
            new_observation, reward, terminated = env.step((agent, adj_schedules_agent[max_action_index]))
                        
            # calculating the Q values in parts 
            alpha = 1 / (1 + num_updates.get((observation, action), 0))
            items = [value for (obs, _), value in Q.items() if obs == action]  
            if len(items) == 0:
                gamma_term = 0
            else:
                gamma_term = gamma * max(items) 
                
            Q.get(observation, {}).get(action, 0) += (alpha * (reward + gamma_term - Q.get(observation, {}).get(action, 0)))
 
            # updating the num_updates matrix 
            num_updates[(observation, action)] = num_updates.get((observation, action), 0) + 1 
            
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
    
    get_adjacent_schedules(sim, lights[0], 5)