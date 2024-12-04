from simulation import Simulation, TrafficLight, LightningMcQueen, Road
from environment import TrafficEnvironment
from graph import Graph, generate_random_path
from itertools import product
from tqdm import tqdm
from util import create_gif_from_images, sanitize_keys
import shutil
import random 
import json
import csv
import gym
import os

"""Cursor cellars (warning: dangerous airborne chemicals, beware of CS major sweat)"""

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
    possible_times = [adjacent_increments(time, step, sim.MAX_LIGHT_TIME) for time in agent.times]
    cartesian_prod = list(product(*possible_times))
    return cartesian_prod


def build_schedule_around(sim: Simulation, agent: TrafficLight, updated_sched: tuple[int]):
    """Builds the full schedule around the single agent's updated schedule"""
    full_sched = (a.get_schedule() if a.id != agent.id else updated_sched for a in sim.agents)
    return full_sched


def control_run(env: gym.Env, num_runs: int, draw_dir=None):
    
    all_congestions = [] 
    all_vehicles = []
    
    # for every run        
    for run in tqdm(range(num_runs)):
        
        congestions = []
        vehicles = [] 
        
        if draw_dir is not None: 
            os.mkdir(f"{draw_dir}/run_{run}")
        
        # reset the environment 
        env.hard_reset()
        
        while env.sim.time < env.sim.MAX_TIME and not env.sim.done(): 
            
            if draw_dir is not None:
                env.sim.draw(f"{draw_dir}/run_{run}")
                                                        
            congestions.append(env.sim.tick())
            vehicles.append(len(env.sim.cars))
            
        print(congestions[-1])
        print(vehicles[-1])
        
        all_congestions.append(congestions)
        all_vehicles.append(vehicles)
                    
        # draw this run and delete the images
        if draw_dir is not None:
            create_gif_from_images(f"{draw_dir}/run_{run}", f"runs/run_{run}.gif", duration=100)
            shutil.rmtree(f"{draw_dir}/run_{run}")
            
    return all_congestions, all_vehicles

        
def q_learning(env: gym.Env, num_episodes: int, time_increments: int = 5, gamma=0.9, epsilon=0.9, draw_dir=None):

    Q = {}
    num_updates = {}
    
    # for every episode        
    for episode in tqdm(range(num_episodes)):
        
        congestions = []
        
        if draw_dir is not None: 
            os.mkdir(f"{draw_dir}/episode_{episode}")
        
        # reset the environment 
        observation = env.hard_reset()
        terminated = False
        agent_index = 0
        
        # while not an ending state
        while not terminated:
            
            if draw_dir is not None:
                env.sim.draw(f"{draw_dir}/episode_{episode}")
                                                        
            prob = random.uniform(0, 1)
            
            # get the agent whose schedule to change in this iteration
            agent = env.sim.agents[agent_index]
            # get all the adjacent schedules for this agent, so the next possible schedules to assign this agent.
            adj_schedules_agent = get_adjacent_schedules(env.sim, agent, time_increments)
            adj_schedules_all = [build_schedule_around(env.sim, agent, sched) for sched in adj_schedules_agent]
                        
            # select action 
            if prob < epsilon:
                action_index = random.randint(0, len(adj_schedules_all) - 1)
                agent_action = adj_schedules_agent[action_index]
                action = tuple(adj_schedules_all[action_index])
            else:
                max_value = 0
                max_action_index = random.randint(0, len(adj_schedules_all) - 1)
                for i, a in enumerate(adj_schedules_all):
                    value = Q.get(observation, {}).get(a, 0)
                    if value > max_value:
                        max_action_index = i
                        max_value = value
                agent_action = adj_schedules_agent[max_action_index]
                action = tuple(adj_schedules_all[max_action_index])                
                
            # get the new observation                    
            new_observation, reward, terminated = env.step((agent, agent_action))
                                                
            # calculating the Q values in parts 
            alpha = 1 / (1 + num_updates.get((observation, action), 0))
            
            actions = Q.get(new_observation, {})
            
            if len(actions) == 0:
                gamma_term = 0
            else:
                gamma_term = gamma * max(actions.values()) 
            
            if observation not in Q:
                Q[observation] = {}
                            
            Q[observation][action] = Q.get(observation, {}).get(action, 0) + (alpha * (reward + gamma_term - Q.get(observation, {}).get(action, 0)))

            # updating the num_updates matrix 
            num_updates[(observation, action)] = num_updates.get((observation, action), 0) + 1 
            
            # updating observation
            observation = new_observation
            
            agent_index = (agent_index + 1) % len(env.sim.agents)
            
                    
        # update epsilon at the end of the episode 
        epsilon = 0.9999 * epsilon 
        
        # draw this episode and delete the images
        if draw_dir is not None:
            create_gif_from_images(f"{draw_dir}/episode_{episode}", f"episodes/episode_{episode}.gif", duration=100)
            shutil.rmtree(f"{draw_dir}/episode_{episode}")
            
        congestions.append(-reward) # add the final congestion of this episode into the list of congestions 
        print(f"Congestion: {-reward}")
        
    # set the optimal policy 
    policy = {}
    for observation in Q:
        # find the optimal action for this observation
        actions = Q.get(observation, {})
        if len(actions) == 0:
            policy[observation] = None
        else:
            best_action = max(actions, key=actions.get)
            policy[observation] = best_action
            
    return Q, policy, congestions


def control():
    
    random.seed(24)

    TOTAL_TIME = 200
    
    NUM_NODES = 10
    NUM_EDGES = 20
    NUM_PATHS = 180
    NUM_PARTITIONS = 4
    MAX_LIGHT_TIME = 30
    NUM_RUNS = 50
    TIME_RATIO = 1
    
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
    
    congestions, vehicles = control_run(env, NUM_RUNS, "graphs/control")
    
    with open("control_congestion.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(congestions)
            
    with open("control_vehicles.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(vehicles)
            

def learning():
        
    random.seed(24)

    TOTAL_TIME = 200
    
    NUM_NODES = 10
    NUM_EDGES = 20
    NUM_PATHS = 140
    NUM_PARTITIONS = 4
    MAX_LIGHT_TIME = 30
    NUM_EPISODES = 10
    DELTA_T = 5
    TIME_RATIO = 1
    
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
        
    Q, policy, congestions = q_learning(env, NUM_EPISODES, DELTA_T, draw_dir="graphs")
    
    # sanitize the keys since json.dumps can't handle tuple as keys, and then write to json
        
    sanitized_Q = sanitize_keys(Q)
    sanitized_policy = sanitize_keys(policy)
    
    with open("Q.json", "w") as file:
        json.dump(sanitized_Q, file, indent=4) 
        
    with open("policy.json", "w") as file:
        json.dump(sanitized_policy, file, indent=4)
        
    with open("control_congestion.txt", "w") as file:
        file.write(" ".join(map(str, congestions)) + "\n") # TODO: THIS IS PROBABLY NOT CORRECT 
    
    
if __name__ == "__main__":
    control()