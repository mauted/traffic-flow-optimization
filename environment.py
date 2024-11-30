import gym
import random
from gym import spaces
from simulation import Simulation, Road, LightningMcQueen, TrafficLight
from graph import Graph, generate_random_path


class TrafficEnvironment(gym.Env):
    
    def __init__(self, sim: Simulation, time_ratio: int):
        
        super(TrafficEnvironment, self).__init__()
        
        self.sim = sim
        # this is the number of ticks the simulations ticks for, for every env tick
        # this ensures that each change of the schedule has some effect before the environment is changed again
        self.time_ratio = time_ratio 
        
        self.max_congestion = sim.INITIAL_NUM_CARS + 1 
        self.num_agents = len(sim.agents)
        self.num_partitions = sim.NUM_PARTITIONS
       
        # corresponding to incoming cars to a traffic light, outgoing cars to a traffic light and the current number of vehicles at the 
        agent_obs_space = spaces.MultiDiscrete([self.max_congestion, self.max_congestion])
        self.observation_space = spaces.Dict({
            agent.id: agent_obs_space for agent in self.sim.agents
        })

        # action space corresponds to the possible schedules per agent, this value is chosen via step and then multiplied by 10
        # to match the actual times desired
        agent_act_space = spaces.MultiDiscrete([int(self.sim.MAX_LIGHT_TIME)] * self.num_partitions)
        self.action_space = spaces.Dict({
            agent.id: agent_act_space for agent in self.sim.agents
        })        

    def reset(self):
        """Reset the environment to an initial state."""
        return self.sim.reset()

    def step(self, agent_action):
        
        agent, action = agent_action
        
        agent.set_schedule(action)
        
        # make time_ratio simulation steps, for the first n - 1 steps we don't care about the congestion
        # for the last step, record what the congestion was
        for _ in range(self.time_ratio - 1):
            self.sim.tick()
        congestion = self.sim.tick()
            
        # now calculate new observations, rewards and done
    
        # observations are per agent, which are done inside the simulation
        observation = self.sim.observation()
        reward = -congestion
        done = self.sim.done()
        
        return observation, reward, done

if __name__ == "__main__":
    
    random.seed(0)

    TOTAL_TIME = 100
    
    NUM_NODES = 3
    NUM_EDGES = 10
    NUM_PATHS = 5
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
    obs = env.reset()
    for i in range(100):
        random_action = env.action_space.sample()
        new_obs, reward, terminated = env.step(random_action)
    
        
