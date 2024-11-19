import gym
from gym import spaces
import numpy as np

class TrafficEnvironment(gym.Env):
    
    def __init__(self):
        
        super(TrafficEnvironment, self).__init__()
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)        
        self.state = None

    def reset(self):
        """Reset the environment to an initial state."""
        self.state = np.random.uniform(low=0, high=1, size=(4,))
        return self.state  # Return the initial observation

    def step(self, action):
        """
        Apply the action to the environment.
        Return:
        - observation: the new state
        - reward: the reward from the action
        - done: whether the episode has ended
        - info: additional info (optional)
        """
        # Example logic:
        self.state = np.random.uniform(low=0, high=1, size=(4,))
        reward = action  # Example reward logic
        done = np.random.rand() > 0.95  # Randomly end episode
        info = {}  # Additional info
        
        return self.state, reward, done, info

    def render(self, mode='human'):
        """Optional: Visualize the environment."""
        print(f"State: {self.state}")

    def close(self):
        """Optional: Cleanup resources."""
        pass
