import gym
from gym import spaces
import numpy as np
from simulation import Simulation

class TrafficEnvironment(gym.Env):
    
    def __init__(self, sim: Simulation):
        
        super(TrafficEnvironment, self).__init__()
        
        self.sim = sim
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32) 
        self.state = None

    def reset(self):
        """Reset the environment to an initial state."""
        return self.sim.reset()

    def step(self, action):
        """
        Apply the action to the environment.
        Return:
        - observation: the new state
        - reward: the reward from the action
        - done: whether the episode has ended
        - info: additional info (optional)
        """
        
        self.sim.update_schedule(action)
        observation = self.sim.observation()
        reward = self.sim.tick()
        done = self.sim.done()
        
        return observation, reward, done, None

    def render(self, mode='human'):
        """Optional: Visualize the environment."""
        print(f"State: {self.state}")

    def close(self):
        """Optional: Cleanup resources."""
        pass
