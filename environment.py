import gym
from gym import spaces
import numpy as np
from simulation import Simulation

## TODO: FORGET ABOUT PERIOD, JUST GENERATE A RANDOM NUMBER 0-100 INCLUSIVE
class TrafficEnvironment(gym.Env):
    
    def __init__(self, sim: Simulation):
        
        super(TrafficEnvironment, self).__init__()
        
        self.sim = sim
        self.max_congestion = sim.INITIAL_NUM_CARS + 1 
        self.num_agents = len(sim.agents)
        self.num_partitions = sim.NUM_PARTITIONS
       
        # corresponding to incoming cars to a traffic light, outgoing cars to a traffic light and the current number of vehicles at the 
        agent_obs_space = spaces.MultiDiscrete([self.max_congestion, self.max_congestion, self.max_congestion])
        self.observation_space = spaces.Dict({
            agent.id: agent_obs_space for agent in self.sim.agents
        })

        # action space corresponds to the possible schedules per agent, this value is chosen via step and then multiplied by 10
        # to match the actual times desired
        agent_act_space = spaces.MultiDiscrete([self.sim.MAX_LIGHT_TIME] * self.num_partitions)
        self.action_space = spaces.Dict({
            agent.id: agent_act_space for agent in self.sim.agents
        })        

    def reset(self):
        """Reset the environment to an initial state."""
        return self.sim.reset()

    def step(self, actions):

        # apply the action to each of the traffic light boys
        for agent_id, action in actions.items():
            agent = self.sim.id_to_agent(agent_id)
            agent.set_schedule(action)
        
        # make 1 simulation step
        congestion = self.sim.tick()
            
        # now calculate new observations, rewards and dones
    
        # observations are per agent, which are done inside the simulation
        observation = self.sim.observation()
        rewards = -congestion
        dones = self.sim.done()
        
        return observation, rewards, dones
